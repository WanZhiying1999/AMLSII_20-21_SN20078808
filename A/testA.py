import numpy as np
import pandas as pd
import nltk
import json
from nltk.tokenize import TweetTokenizer
from keras.models import load_model
from keras.utils.np_utils import*
from sklearn.metrics import f1_score

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
        X_test[i, j] = index

def test_A(Clean_testA):
    #load tidy data
    data = pd.read_csv(Clean_testA)
    #remain label and tidy tweet
    data = data[['Sentiment','tidy_tweet']]
    #creat dataset and tokenization
    tk = TweetTokenizer(reduce_len=True)
    global X_test,Y_test
    X_test = []
    Y_test = []
    tidy_token_list = []
    for x, y in zip(data['tidy_tweet'], data['Sentiment']):
        x=json.dumps(x)
        X_test.append(tk.tokenize(x))
        Y_test.append(y)
        tidy_token_list.append((tk.tokenize(x), y))

    #print(tidy_token_list)
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('model\\glove.6B.100d.txt') 


    max_len = 30
    print('max_len:', max_len)

    X_test = np.zeros((len(tidy_token_list), max_len))
    Y_test = np.zeros((len(tidy_token_list), ))

    for i, tk_lb in enumerate(tidy_token_list):
        tokens, label = tk_lb
        sentence_to_indices(tokens, word_to_index, max_len, i)
        Y_test[i] = label
    
    Y_test = to_categorical(Y_test, 3)


    model = load_model('A\\modelA.hdf5')
    y_pred = model.predict(X_test)
    score = model.evaluate(X_test, Y_test)

    y_pred = np.argmax(y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    #print(y_pred)

    f1 = f1_score(Y_test, y_pred, average='weighted')


    return score[1],f1
#acc_A_test = test_A('Datasets\\Test\\test_A.csv')
#print(acc_A_test)