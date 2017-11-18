# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:13:38 2017

@author: NishitP

Note: before we can train an algorithm to classify fake news labels, we need to extract features from it. It means reducing the mass
of unstructured data into some uniform set of attributes that an algorithm can understand. For fake news detection, it could be 
word counts (bag of words). 

"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import gensim
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score


#from collection import counter
from DataPrep.py import *
import numpy as np
import pandas as pd


#file read (needs to be removed in final version)
test_filename = 'test.csv'
train_filename = 'train.csv'

train_news = pd.read_csv(train_filename)
test_news = pd.read_csv(test_filename)


#we will start with simple bag of words technique 
#creating feature vector - document term matrix
countV = CountVectorizer()
train_count = countV.fit_transform(train_news['Statement'])

#print training doc term matrix
#we have matrix of size of (10240, 12196) by calling below
train_count.shape

#check vocabulary using below command
print(countV.vocabulary_)

#get feature names
print(countV.get_feature_names()[:25])


#building classifier using naive bayes 
nb_pipeline = Pipeline([
        ('NBCV',countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(train_news['Statement'],train_news['Label'])

doc_new = ['obama is running for president in 2016']

nb_pipeline.predict(doc_new)
predicted_nb = nb_pipeline.predict(test_news['Statement'])
np.mean(predicted_nb == test_news['Label'])
#accuracy = 0.61


#building classifier using logistic regression
logR_pipeline = Pipeline([
        ('LogRCV',countV),
        ('LogR_clf',LogisticRegression())
        ])

logR_pipeline.fit(train_news['Statement'],train_news['Label'])

logR_pipeline.predict(doc_new)
predicted_LogR = logR_pipeline.predict(test_news['Statement'])
np.mean(predicted_LogR == test_news['Label'])
#accuracy = 0.61


#building SVM classfier
svm_pipeline = Pipeline([
        ('svmCV',countV),
        ('svm_clf',svm.SVC())
        ])

svm_pipeline.fit(train_news['Statement'],train_news['Label'])

svm_pipeline.predict(doc_new)
predicted_svm = svm_pipeline.predict(test_news['Statement'])
np.mean(predicted_svm == test_news['Label'])
#accuracy = 0.56

#using svm again
svm2_pipeline = Pipeline([
        ('svm2CV',countV),
        ('svm2_clf',SGDClassifier())
        ])

svm2_pipeline.fit(train_news['Statement'],train_news['Label'])


predicted_svm2 = svm2_pipeline.predict(test_news['Statement'])
np.mean(predicted_svm2 == test_news['Label'])
#accurary = 0.58


#User defined functon for K-Fold cross validatoin
def build_confusion_matrix(classifier):
    
    k_fold = KFold(n=len(train_news), n_folds=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold:
        train_text = train_news.iloc[train_ind]['Statement'] 
        train_y = train_news.iloc[train_ind]['Label']
    
        test_text = train_news.iloc[test_ind]['Statement']
        test_y = train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    
    return (print('Total statements classified:', len(train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('Confusion matrix:'),
    print(confusion))
    
#K-fold cross validation for all classifiers
build_confusion_matrix(nb_pipeline)
build_confusion_matrix(logR_pipeline)
build_confusion_matrix(svm_pipeline)
build_confusion_matrix(svm2_pipeline)

#========================================================================================
#Bag of words confusion matrix and F1 scores

#Naive bayes
# [2118 2370]
# [1664 4088]
#0.669611539651

#Logistic regression
# [2252 2236]
# [1933 3819]
# 0.646909097798

#svm1
# [0 4488]
# [ 0 5752]
# 0.719313230705

#svm2 - sgdclassifier
# [2535 1953]
# [2502 3250]
# 0.590245654241
#=========================================================================================
    



"""Usinng n-grams, stopwords etc """
##Now using n-grams
nb_pipeline_ngram = Pipeline([
        ('NBCV',CountVectorizer(ngram_range=(1,3),stop_words='english')),
        ('tfidf',TfidfTransformer(use_idf=True,smooth_idf=True)),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(train_news['Statement'],train_news['Label'])

doc_new = ['trump is running for president in 2016']

predicted_nb_ngram = nb_pipeline_ngram.predict(test_news['Statement'])
np.mean(predicted_nb_ngram == test_news['Label'])
#accuracy = 0.60


logR_pipeline_ngram = Pipeline([
        ('LogRCV',CountVectorizer(ngram_range=(1,3),stop_words='english')),
        ('tfidf',TfidfTransformer(use_idf=True,smooth_idf=True)),
        ('LogR_clf',LogisticRegression())
        ])

logR_pipeline_ngram.fit(train_news['Statement'],train_news['Label'])

predicted_LogR = logR_pipeline_ngram.predict(test_news['Statement'])
np.mean(predicted_LogR == test_news['Label'])
#accuracy = 0.62


svm_pipeline_ngram = Pipeline([
        ('svmCV',CountVectorizer(ngram_range=(1,3),stop_words='english')),
        ('tfidf',TfidfTransformer(use_idf=True,smooth_idf=True)),
        ('svm_clf',svm.SVC())
        ])

svm_pipeline_ngram.fit(train_news['Statement'],train_news['Label'])

predicted_svm = svm_pipeline_ngram.predict(test_news['Statement'])
np.mean(predicted_svm == test_news['Label'])
#accuracy = 0.56


svm2_pipeline_ngram = Pipeline([
         ('svm2CV',CountVectorizer(ngram_range=(1,3),stop_words='english')),
         ('tfidf',TfidfTransformer(use_idf=True,smooth_idf=True)),
         ('svm2_clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, n_iter=5))
         ])

svm2_pipeline_ngram.fit(train_news['Statement'],train_news['Label'])

svm2_pipeline_ngram.predict(doc_new)

predicted_svm2 = svm2_pipeline_ngram.predict(test_news['Statement'])
np.mean(predicted_svm2 == test_news['Label'])
#accurary = 0.57


#K-fold cross validation for all classifiers
build_confusion_matrix(nb_pipeline_ngram)
build_confusion_matrix(logR_pipeline_ngram)
build_confusion_matrix(svm_pipeline_ngram)
build_confusion_matrix(svm2_pipeline_ngram)


#========================================================================================
#n-grams & tfidf confusion matrix and F1 scores

#Naive bayes
# [ 841 3647]
# [ 427 5325]
# 0.723262051071

#Logistic regression
# [1617 2871]
# [1097 4655]
# 0.70113000531

#svm1
# [0 4488]
# [ 0 5752]
# 0.719313230705

#svm2 - sgdclassifier
# [  10 4478]
# [  13 5739]
# 0.718731637053
#=========================================================================================



#grid-search parameter optimization
parameters = {'vect__ngram_range': [(1, 1)], #(1, 2),(1,3)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(nb_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train_news['Statement'],train_news['Label'])


gs_clf.best_score_
gs_clf.best_params_



"""Usinng Word2Vec """
model = gensim.models.Word2Vec(X, size=100) # x be tokenized text
w2v = dict(zip(model.wv.index2word, model.wv.syn0))


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


svm2_pipeline_w2v = Pipeline([
         ('w2v',MeanEmbeddingVectorizer(w2v)),
         ('svm2_clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, n_iter=5))
         ])

svm2_pipeline_w2v_tfidf = Pipeline([
         ('w2v',TfidfEmbeddingVectorizer(w2v)),
         ('svm2_clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, n_iter=5))
         ])



#tf-idf 
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)
train_tfidf.shape

#get feature names
print(tfidfV.get_feature_names()[:25])


#get train data feature names 
print(train_tfidf.A[:10])


#build classifier
#naive bayes
nb_clf = MultinomialNB().fit(train_tfidf,train_news['Label'])

docs_new = ["hillary is running for president", "hillary clinton is running for president","obama was born in kenya"]
train_new_count = countV.transform(docs_new)
train_new_tfidf = tfidfV.transform(train_new_count)

pred = nb_clf.predict(train_new_tfidf)

for doc, category in zip(docs_new, pred):
     print('%r => %s' % (doc, pred))

print(docs_new,pred)





