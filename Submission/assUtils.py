import pandas as pd
import numpy as np
import spacy
import re
from collections import Counter
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


def text_preprocesser(data, col_name):  # fiton specifies what data to fit the pre processor on
    documents = data[col_name]

    # removing numbers
    documents = documents.str.replace(r'\d+', '') # matches one or more digits followed by a slash followed by one or more digits.

    # Make the review data Lower case
    documents = documents.str.lower() # turns the text to lower case

    # Filtering out stop words
    vectorizer = CountVectorizer(stop_words='english') # loading the stop words
    tokenize = vectorizer.build_tokenizer()
    documents = documents.apply(tokenize) # removing the stop words from each row in the text column

    # Stemming the text data
    stemmer = SnowballStemmer('english')  # snowball stemmer is used to reduce words
    documents = documents.apply(lambda x: [stemmer.stem(y) for y in x])

    # Lemmatizing the data
    lemmatizer = WordNetLemmatizer()
    documents = documents.apply(lambda x: [lemmatizer.lemmatize(y, 'v') for y in x])
    documents = documents.apply(lambda x: ' '.join([str(elm) for elm in x]))

    return documents

def labels_to_number(df):
    labels = list(set(df['label']))
    labels.sort()
    label_dict = {label: indx for indx, label in enumerate(labels)}  # getting the labels for classification in alphabetical order and storing them in a dictionary
    df.replace({'label' : label_dict} , inplace = True)

tok = spacy.load('en_core_web_sm') # loading spacy tokenizers
def tokenize(text):

    text = re.sub(r"[^\x00-\x7F]+", " ", text) # removing
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

def tweet_clean(df):
    df['tweet_text'] = df['tweet_text'].fillna('') # replacing NA with blanks
    df['tweet_length'] = df['tweet_text'].apply(lambda x: len(x.split())) # generating a new column, which stores the number of words in a tweet

    labels_to_number(df)
    return df

def build_vocabulary(df):
    counts = Counter()
    for index, row in df.iterrows():
        counts.update(tokenize(row['tweet_text']))

    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]

    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    return vocab2index, words, counts

def encode_sentence_cuda(text, vocab2index, N = 40): # NB that this N is important for speed
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype = int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    return torch.as_tensor(encoded, dtype = torch.long, device = device), length

def encode_sentence(text, vocab2index, N = 40): # NB that this N is important for speed
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype = int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length

def encode_df(df, vocab2index):
    df['encoded'] = df['tweet_text'].apply(lambda x: encode_sentence(x, vocab2index))


class ReviewsDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]

class ReviewsDatasetCuda(Dataset):

    def __init__(self, X, Y):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        self.X = X
        self.y = torch.as_tensor(Y, dtype = torch.long, device = device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx][0], self.y[idx], self.X[idx][1]

def load_embedding_vectors(filename = 'glove.6B.50d.txt'):
    '''Load Word Vectors'''
    word_vectors = {}
    with open(filename, encoding = 'utf8') as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    return word_vectors

def get_emb_matrix(pretrained, word_counts, emb_size = 200):
    '''Get the embedding matrix from word vectors'''
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype = "float32")
    W[0] = np.zeros(emb_size, dtype = 'float32') # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in pretrained:
            W[i] = pretrained[word]
        else:
            W[i] = np.random.uniform(-0.25, 0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1
    return W, np.array(vocab), vocab_to_idx


class LSTM_pretrained_embeddings(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_weights, num_layers=1, bidirectional=True, dropout_rate=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_weights.shape[1]
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, self.hidden_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self.embeddings.weight.data.copy_(torch.from_numpy(embedding_weights).to(device))
        self.embeddings.weight.requires_grad = False  ## Prevent autograd changing the weights
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=bidirectional,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim * 2, 3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)

        #packed_embedded = nn.utils.rnn.pack_padded_sequence(x, l, batch_first = True, enforce_sorted=False)
        #lstm_out, (ht, ct) = self.lstm(packed_embedded)

        lstm_out, (ht, ct) = self.lstm(x)
        ht = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)

        return self.linear(ht)


class GRU_pretrained_embeddings(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_weights, num_layers=1, bidirectional=True, dropout_rate=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_weights.shape[1]
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, self.hidden_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        self.embeddings.weight.data.copy_(torch.from_numpy(embedding_weights).to(device))
        self.embeddings.weight.requires_grad = False  ## Prevent autograd changing the weights
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=bidirectional,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim * 2, 3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)

        gru_out, ht= self.gru(x)
        ht = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)

        return self.linear(ht)


def plot_metric(metrics_df, metric = "acc", title = "Accuracy vs Epoch", name = "test"):
    import matplotlib as mpl
    mpl.style.use('default')
    import matplotlib.pyplot as plt
    plt.plot(metrics_df[['train_' + metric]], label = "train " + metric)
    plt.plot(metrics_df[['valid_' + metric]], label = "test " + metric)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig("Figures/" + name + ".png")
    plt.show()

def train_model(mod, config, vocab_size, pretrained_weights, train_ds, valid_ds):

    global y_true_train_store
    train_dl = DataLoader(train_ds, batch_size=int(config['batch_size']), shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=int(config['batch_size']), shuffle=True)

    model = mod(
        vocab_size,
        hidden_dim=config['hidden_dim'],
        embedding_weights=pretrained_weights,
        num_layers=config['num_layers'],
        bidirectional=True,
        dropout_rate=config['dropout_rate'])

    device = "cuda"
    if torch.cuda.is_available():
        device = "cuda"
    model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(parameters, lr=config['lr'], weight_decay=config['weight_decay'],
                                 amsgrad=config['amsgrad'])

    epochs = config['epochs']

    train_loss_store = []
    train_acc_store = []
    valid_loss_store = []
    valid_acc_store = []

    for epoch in range(epochs):

        y_true_train_store = []
        y_pred_train_store = []
        y_true_test_store = []
        y_pred_test_store = []

        print("Total Completed",epoch/epochs)
        model.train()
        running_loss_train = 0.0
        train_steps = 0
        total_train = 0
        correct_train = 0

        for x_train, y_train, l_train in train_dl:
            x_train = x_train.long().to(device)
            y_train = y_train.long().to(device)

            y_pred_train = model(x_train, l_train)  # getting the predictions
            optimizer.zero_grad() # gradients must be reset to 0 for each iteration, pytorch keeps them by default
            loss = criterion(y_pred_train, y_train)
            loss.backward()
            optimizer.step()

            pred = torch.max(y_pred_train, 1)[1]

            if epoch + 1 == epochs:
                y_pred_train_store += pred.squeeze().tolist()
                y_true_train_store += y_train.squeeze().tolist()

            correct_train += (pred == y_train).float().sum()
            total_train += y_train.shape[0]

            running_loss_train += loss.item()
            train_steps += 1

        train_loss_store.append(running_loss_train/train_steps)
        train_acc_store.append((correct_train/total_train).cpu().item())

        model.eval()
        running_loss_valid = 0.0
        valid_steps = 0
        total_valid = 0
        correct_valid = 0

        for x_valid, y_valid, l_valid in valid_dl:
            with torch.no_grad():
                x_valid = x_valid.long().to(device)
                y_valid = y_valid.long().to(device)
                y_pred_valid = model(x_valid, l_valid)

                pred = torch.max(y_pred_valid, 1)[1]

                if epoch + 1 == epochs:
                    y_pred_test_store += pred.cpu().squeeze().tolist()
                    y_true_test_store += y_valid.cpu().squeeze().tolist()

                correct_valid += (pred == y_valid).float().sum()
                total_valid += y_valid.shape[0]

                loss = criterion(y_pred_valid, y_valid)
                running_loss_valid += loss.cpu().numpy()
                valid_steps += 1

        valid_loss_store.append(running_loss_valid/valid_steps)
        valid_acc_store.append((correct_valid/total_valid).cpu().item())

    metrics = [train_loss_store, train_acc_store, valid_loss_store, valid_acc_store]
    metrics = pd.DataFrame(metrics).transpose()
    metrics.columns = ["train_loss", "train_acc", "valid_loss", "valid_acc"]

    ys = [y_true_train_store, y_pred_train_store, y_true_test_store, y_pred_test_store]

    ys = pd.DataFrame(ys).transpose()
    ys.columns = ["y_true_train", "y_pred_train", "y_true_test", "y_pred_test"]


    path = "C:/Users/david/Documents/Masters/Semester2/COMP47650_Deep_Learning/Assignment/Notebooks/model.pth"
    # torch.save(model,path)
    return metrics, ys #, model.parameters()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    np.set_printoptions(precision=2)
    from sklearn.utils.multiclass import unique_labels
    import matplotlib as mpl
    mpl.style.use('classic')

    from sklearn.metrics import confusion_matrix
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    fig.savefig("Figures/" + title + ".png")
    return ax


def results(y_true, y_pred, model_type="LSTM"):

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    print(model_type + " " + "Accuracy", accuracy_score(y_true, y_pred))
    print(model_type + " " + "Precision", precision_score(y_true, y_pred, average="macro"))
    print(model_type + " " + "Recall", recall_score(y_true, y_pred, average="macro"))
    print(model_type + " " + "F1 Score", f1_score(y_true, y_pred, average="macro"))

    plot_confusion_matrix(y_true, y_pred, classes=['Angry', 'Dissapointed', "Happy"], normalize=True,
                          title= model_type + '_Confusion_Matrix');