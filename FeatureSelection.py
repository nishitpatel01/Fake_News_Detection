# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:13:38 2017

@author: NishitP

Note: before we can train an algorithm to classify fake news labels, we need to extract features from it. It means reducing the mass
of unstructured data into some uniform set of attributes that an algorithm can understand. For fake news detection, it could be 
word counts (bag of words). 

"""
from DataPrep.py import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models.word2vec import Word2Vec
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score

#from collection import counter


#file read (needs to be removed in final version)
test_filename = 'test.csv'
train_filename = 'train.csv'

train_news = pd.read_csv(train_filename)
test_news = pd.read_csv(test_filename)


#we will start with simple bag of words technique 
#creating feature vector - document term matrix
countV = CountVectorizer(stop_words='english')
train_count = countV.fit_transform(train_news['Statement'])

#print training doc term matrix
#we have matrix of size of (10240, 12196) by calling below
train_count.shape

#check vocabulary using below command
print(countV.vocabulary_)

#get feature names
print(countV.get_feature_names()[:25])

#


#tf-idf 
tfidfV = TfidfTransformer()
train_tfidf = tfidfV.fit_transform(train_count)
train_tfidf.shape


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





#Usinng Word2Vec 
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



"""

