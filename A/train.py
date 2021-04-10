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

#use  pre-trained GloVe(Global vectors for words representations) mdoel to realize word embeddings
# read glove file and output three dictionaries
# * word_to_index: a dictionary mapping from words to their indices in the vocabulary
# * index_to_word: dictionary mapping from indices to their corresponding words in the vocabulary
# * word_to_vec_map: dictionary mapping words to their GloVe vector representation 
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

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

def pretrained_embedding_layer(word_to_vec_map, word_to_index, max_len):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["unk"].shape[0] #
    
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]
        
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False, input_shape=(max_len,))
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def A_train(Clean_trainA):
    #load tidy data
    data = pd.read_csv(Clean_trainA)
    #remain label and tidy tweet
    data = data[['Sentiment','tidy_tweet']]
    #df_pos = data[data['Sentiment'] == 'positive']
    #df_neg = data[data['Sentiment'] == 'negative']
    #df_neu = data[data['Sentiment'] == 'neutral']
    #print(len(df_pos), len(df_neg),len(df_neu))

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

    #print(tidy_token_list)
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('model\\glove.6B.100d.txt') 
    #data padding 
    #calculate the length of the longest phrase
    

    #list_len = [len(i) for i, j in tidy_token_list]
    #max_len = max(list_len)
    max_len = 30
    print('max_len:', max_len)

    X = np.zeros((len(tidy_token_list), max_len))
    Y = np.zeros((len(tidy_token_list), ))

    for i, tk_lb in enumerate(tidy_token_list):
        tokens, label = tk_lb
        sentence_to_indices(tokens, word_to_index, max_len, i)
        Y[i] = label

    Y = to_categorical(Y, 3)

    #print(X[:1])
    #print(Y[:5])

    #X_train, Y_train, X_val, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
    #build the model
    #embedding layer


    model = Sequential()

    model.add(pretrained_embedding_layer(word_to_vec_map, word_to_index, max_len))
    model.add(Dropout(rate=0.4))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(rate=0.4))
    model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
    model.add(Dense(units=3, activation='softmax'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    H = model.fit(X, Y, validation_split=0.2, epochs = 10, batch_size = 32, shuffle=True)
    model.save('A\\modelA.hdf5')

    plt.plot(H.history['acc'])
    plt.plot(H.history['val_acc'])
    plt.title('Model A accuracy')
    plt.ylabel('Accuracy')  
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()
    plt.savefig("Accuracy.png")
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('Model A loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()


    return H.history['acc'][-1], H.history['val_acc'][-1]







