# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:00:49 2017

@author: NishitP
"""
#import os
import pandas as pd
import csv
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb

#set working directory
#os.chdir("C:/Users/NishitP/Desktop/UIUC MCS-DS/CS-410 - Text Information Systems - FALL 2017/Final Project/CS410 - Fake News Detection")

test_filename = 'test.csv'
train_filename = 'train.csv'

train_news = pd.read_csv(train_filename, header=None)
test_news = pd.read_csv(test_filename, header=None)


#data observation
print(train_news.shape)
print(train_news.head(10))

print(test_news.shape)
print(test_news.head(10))

#distribution of classes
sb.countplot(x='Label',data=train_news, palette='hls')

#explanation here

#data integrity check (missing label values)
train_news.isnull().sum()
train_news.info()


eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))


#Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

#process the data
def process_data(data,exclude_stopword=True,stem=True):
    tokens = [w.lower() for w in data]
    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)
    tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords ]
    return tokens_stemmed


#creating ngrams
#unigram 
def create_unigram(words):
    assert type(words) == list
    return words

#bigram
def create_bigrams(words):
    assert type(words) == list
    skip = 0
    join_str = " "
    Len = len(words)
    if Len > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append(join_str.join([words[i],words[i+k]]))
    else:
        #set it as unigram
        lst = create_unigram(words)
    return lst

#trigrams
def create_trigrams(words):
    assert type(words) == list
    skip == 0
    join_str = " "
    Len = len(words)
    if L > 2:
        lst = []
        for i in range(1,skip+2):
            for k1 in range(1, skip+2):
                for k2 in range(1,skip+2):
                    for (i+k1) < Len and (i+k1+k2) < Len:
                        lst.append(join_str.join([words[i], words[i+k1],words[i+k1+k2])])
        else:
            #set is as bigram
            lst = create_bigram(words)
    return lst

test_news = pd.read_csv(test_filename, header=None)

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

doc = ['runners like running and thus they run','this is a test for tokens']
tokenizer([word for line in test_news.iloc[:,1] for word in line.lower().split()])



#show the distribution of labels in the train and test data
"""def create_datafile(filename)
    #function to slice the dataframe to keep variables necessary to be used for classification
    return "return df to be used"
"""
    
"""#converting multiclass labels present in our datasets to binary class labels
for i , row in data_TrainNews.iterrows():
    if (data_TrainNews.iloc[:,0] == "mostly-true" | data_TrainNews.iloc[:,0] == "half-true" | data_TrainNews.iloc[:,0] == "true"):
        data_TrainNews.iloc[:,0] = "true"
    else :
        data_TrainNews.iloc[:,0] = "false"
        
for i,row in data_TrainNews.iterrows():
    print(row)
"""
    
print(data_TrainNews.head(5))
print(type(data_TrainNews.iloc[:,0]))
print(type("true"))