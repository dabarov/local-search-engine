import pandas as pd

from sklearn import linear_model, feature_selection, preprocessing
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from statsmodels.tools.tools import add_constant
from statsmodels.tools.eval_measures import mse
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import numpy as np
import re
from textblob import TextBlob as tb
import math

import numpy as np
import string
import random
import nltk
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer


# loading dataset
# ---
articles1 = pd.read_csv('kaggle-news-dataset/articles1.csv')
articles2 = pd.read_csv('kaggle-news-dataset/articles2.csv')
articles3 = pd.read_csv('kaggle-news-dataset/articles3.csv')

df =  pd.concat([articles1, 
                 articles2, 
                 articles3], ignore_index=True)

raw_number = df.shape[0]
# --- 
exclude = set(string.punctuation)
alldocslist = []

for i in range(raw_number):
    text = df['content'][i]
    text = ''.join(ch for ch in text if ch not in exclude)
    alldocslist.append(text)

print(alldocslist[1])

# %% [code]
# tokenize words in all DOCS
plot_data = [[]] * len(alldocslist)

for doc in alldocslist:
    text = doc
    tokentext = word_tokenize(text)
    plot_data[raw_number - 1].append(tokentext)

print(plot_data[0][1])

# %% [code]
# Navigation: first index gives all documents, second index gives specific document, third index gives words of that doc
plot_data[0][1][0:10]

# %% [code]

# make all words lower case for all docs
for x in range(len(reuters.fileids())):
    lowers = [word.lower() for word in plot_data[0][x]]
    plot_data[0][x] = lowers

plot_data[0][1][0:10]

# %% [code]
# remove stop words from all docs
stop_words = set(stopwords.words('english'))

for x in range(len(reuters.fileids())):
    filtered_sentence = [w for w in plot_data[0][x] if not w in stop_words]
    plot_data[0][x] = filtered_sentence

plot_data[0][1][0:10]

# %% [code]
# stem words EXAMPLE (could try others/lemmers )

snowball_stemmer = SnowballStemmer("english")
stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
stemmed_sentence[0:10]

porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
stemmed_sentence[0:10]

# %% [markdown]
# # PART II: CREATING THE INVERSE-INDEX

# %% [code]
# Create inverse index which gives document number for each document and where word appears

# first we need to create a list of all words
l = plot_data[0]
flatten = [item for sublist in l for item in sublist]
words = flatten
wordsunique = set(words)
wordsunique = list(wordsunique)

# %% [code]
# create functions for TD-IDF / BM25


def tf(word, doc):
    return doc.count(word) / len(doc)


def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc)


def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)))


def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))


# %% [code]
# Create dictonary of words
# THIS ONE-TIME INDEXING IS THE MOST PROCESSOR-INTENSIVE STEP AND WILL TAKE TIME TO RUN (BUT ONLY NEEDS TO BE RUN ONCE)

plottest = plot_data[0][0:1000]

worddic = {}

for doc in plottest:
    for word in wordsunique:
        if word in doc:
            word = str(word)
            index = plottest.index(doc)
            positions = list(np.where(np.array(plottest[index]) == word)[0])
            idfs = tfidf(word, doc, plottest)
            try:
                worddic[word].append([index, positions, idfs])
            except:
                worddic[word] = []
                worddic[word].append([index, positions, idfs])

# %% [code]
# the index creates a dic with each word as a KEY and a list of doc indexs, word positions, and td-idf score as VALUES
worddic['china']

# %% [code]
# pickel (save) the dictonary to avoid re-calculating
np.save('worddic_1000.npy', worddic)