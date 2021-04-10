#define a function to preprocess original tweet
import numpy as np
import pandas as pd
from termcolor import colored
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#expand word n't to not
def expand_tweet(tweet):
	expanded_tweet = []
	for word in tweet:
		if re.search("n't", word):
			expanded_tweet.append(word.split("n't")[0])
			expanded_tweet.append("not")
		else:
			expanded_tweet.append(word)
	return expanded_tweet

#remove URL,numbers, lemmatizing the words
def clean_tweet(tweet, wordNetLemmatizer):
    

    print(colored("Removing tweet mention @user","yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].str.replace("@[\w]*"," ")
    
    print(colored("Removing URLs","yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].replace(re.compile(r'((www\.[^\s]+)|(https?://[^\s]+))'), " ")

    print(colored("Removing numbers and special characters","yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].str.replace("[^a-zA-Z']"," ")
      
    print(colored("Removing words with length less than 3","yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].apply(lambda tweet: ' '.join([w for w in tweet.split() if len(w)>2]))

    print(colored("Tokenizing","yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].apply(lambda tweet: tweet.split())

    print(colored("Removing stop words","yellow"))
    stop_words = set(stopwords.words('english'))
    stop_words.remove("not")
    tweet['tidy_tweet'] = tweet['tidy_tweet'].apply(lambda tweet: [word for word in tweet if word not in stop_words])

    print(colored("Expanding not words","yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].apply(lambda tweet: expand_tweet(tweet))

    print(colored("Lemmatizing the words", "yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].apply(lambda tweet: [wordNetLemmatizer.lemmatize(word) for word in tweet])

    print(colored("Combining words back to tweets", "yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].apply(lambda tweet: ' '.join(tweet))

    print(colored("Converting to lower case","yellow"))
    tweet['tidy_tweet'] = tweet['tidy_tweet'].str.lower()

    return tweet


def Tweet_preprocessing(OriginalTweet,CleanTweet):
    print(colored("Loading data...","yellow"))
    data = pd.read_csv(OriginalTweet,delimiter='\t',header=None,names=['ID','Sentiment','tweet','NAN'])
    
    print(colored("Cleaning data...","yellow"))
    data = data.drop_duplicates().reset_index(drop=True) 
    data['tidy_tweet'] = data['tweet']
    data = data.drop(['ID'],axis=1)
    data = data.drop(['NAN'],axis=1)

    wordNetLemmatizer = WordNetLemmatizer()
    data = clean_tweet(data, wordNetLemmatizer)
    #convert negative to 0, neutral to 1, positive to 2
    data['Sentiment'] = data['Sentiment'].map({'negative':0, 'neutral':1, 'positive':2})
    #print(data)
    data.to_csv(CleanTweet,index=False)

#Tweet_preprocessing('Datasets\\SemEval2017-task4-dev.subtask-A.english.INPUT.txt','Datasets\\Train_A.csv')
