# Fake News Detection

Fake News Detection in Python

This project is part of CS410:Text Information System course. We have used various natural language processing and machine learning libaries from python. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them:

#### Dataet used
The data source used for this project is from LIAR dataset which has 3 files with .tsv format for test, train and validation. Below is some description about the data files used for this project.
	
LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.

Description of the TSV format:

Column 1: the ID of the statement ([ID].json).
Column 2: the label. (Label class contains: True, Mostly-true, Half-true, Barely-true, FALSE, Pants-fire)
Column 3: the statement.
Column 4: the subject(s).
Column 5: the speaker.
Column 6: the speaker's job title.
Column 7: the state info.
Column 8: the party affiliation.
Column 9-13: the total credit history count, including the current statement.
9: barely true counts.
10: false counts.
11: half true counts.
12: mostly true counts.
13: pants on fire counts.
Column 14: the context (venue / location of the speech or statement).

### File descriptions

#### DataPrep.py
This file contains all the pre processing function needed to process all input documents. First we read the train, test and validation data files then some pre processing like tokenizing, stemming etc. There are some exploratory data analysis is performed like prediction class distribution and data quality checks like null or missing values.

#### FeatureSelection.py
In this file we have used feature extraction and selection methods from sci-kit learn python libraries. for feature selection, we have used methods like simple bag-of-words and n-grams and then term frequency like tf-tdf weighting. we have also used word2vec and POS tagging to extract the feature.

#### classifier.py
Here we have build all the classifiers for predicting the fake news detection. The extracted features are fed to different classfier. We have used Naive-bayes, Logistic Regression, Linear SVM, Stochastic gradient decent and Random forest. Each of the extracted featues were used in all of the classifiers. Once fitting the model, we compared the f1 score and checked the confusion matrix. After fitting all the classifiers, 2 best peforming models were selected as candidate models for fake news classification. Finally selected model was used for fake news detection with the probability of truth.

#### prediction.py
This file used the saved classification model to classify the news article from use. It takes an news article as input from user then model is used for final clasdification output that is shown to user along with probability of truth.


### Installing

A step by step series of examples that tell you have to get a development env running

1. The first step would be to clone this repo to your local machine. To do that you need to run following command in commanda prompt or in git bash
```
$ git clone https://github.com/nishitpatel01/fake_news_detection_dev.git
```

2. This will copy all the data source file, program files and model into your machine.

3. After all the files are saved in a folder in your machine. open the git terminal and type below command and press enter.
```
python prediction.py
```

4. After hitting the enter, program will use for an input string which will be a piece of information or a news headline that you want to verify. Once you paste or type news headline, then press enter.


End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```


## Contributing



## Versioning


