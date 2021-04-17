import numpy as np
import pandas as pd
import json
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import*
import keras
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

 
def read_glove(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec[curr_word] = np.array(line[1:], dtype=np.float64)
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec

def cleared(word):
    res = ""
    prev = None
    for char in word:
        if char == prev: continue
        prev = char
        res += char
    return res

def sentence_to_indices(sentence_words, word_to_index, max_len, i):
    
    sentence_indices = []
    unks = []
    UNKS = []
    for j, w in enumerate(sentence_words):
        try:
            index = word_to_index[w]
        except:
            UNKS.append(w)
            w = cleared(w)
            try:
                index = word_to_index[w]
            except:
                index = word_to_index['unk']
                unks.append(w)
        X[i, j] = index

def pretrained_embedding_layer(word_to_vec, word_to_index, max_len):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec["unk"].shape[0] 
    
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec[word]
        
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False, input_shape=(max_len,))
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def bilstm(Clean_trainA):
    #load tidy data
    data = pd.read_csv(Clean_trainA)
    #remain label, topic and tidy tweet
    data = data[['Sentiment','tidy_tweet']]

    #creat dataset and tokenization
    tk = TweetTokenizer(reduce_len=True)
    global X, Y
    X = []
    Y = []
    tidy_token_list = []
    for x, y in zip(data['tidy_tweet'], data['Sentiment']):
        x=json.dumps(x)
        X.append(tk.tokenize(x))
        Y.append(y)
        tidy_token_list.append((tk.tokenize(x), y))

    word_to_index, index_to_word, word_to_vec = read_glove('model\\glove.6B.100d.txt') 
    
    list_len = [len(i) for i, j in tidy_token_list]
    max_len = max(list_len)
    #max_len = 50
    print('max_len:', max_len)

    X = np.zeros((len(tidy_token_list), max_len))
    Y = np.zeros((len(tidy_token_list), ))

    for i, tk_lb in enumerate(tidy_token_list):
        tokens, label = tk_lb
        sentence_to_indices(tokens, word_to_index, max_len, i)
        Y[i] = label

    Y = to_categorical(Y, 3)

    model = Sequential()

    model.add(pretrained_embedding_layer(word_to_vec, word_to_index, max_len))
    model.add(Dropout(rate=0.4))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(rate=0.4))
    model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
    model.add(Dense(units=3, activation='softmax'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    H = model.fit(X, Y, validation_split=0.2, epochs = 10, batch_size = 64, shuffle=True)


    return H.history['acc'][-1], H.history['val_acc'][-1]

acc_A_train, val_acc_A_train = bilstm('Datasets\\Train\\train_A.csv')
print(acc_A_train)
print(val_acc_A_train)










