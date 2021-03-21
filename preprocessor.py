import math
import nltk
import re
import string

import numpy as np
import pandas as pd

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer


def tf(word, doc):
    return doc.count(word) / len(doc)


def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc)


def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)))


def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))


articles1 = pd.read_csv('kaggle-news-dataset/articles1.csv')
articles2 = pd.read_csv('kaggle-news-dataset/articles2.csv')
articles3 = pd.read_csv('kaggle-news-dataset/articles3.csv')

df = pd.concat([articles1,
                articles2,
                articles3], ignore_index=True)

raw_number = df.shape[0]
exclude = set(string.punctuation)
alldocslist = []

for i in range(raw_number):
    text = df['content'][i]
    text = ''.join(ch for ch in text if ch not in exclude)
    alldocslist.append(text)

print(alldocslist[1])

plot_data = [[]] * len(alldocslist)

for doc in alldocslist:
    text = doc
    tokentext = word_tokenize(text)
    plot_data[raw_number - 1].append(tokentext)

print(plot_data[0][1])

print(plot_data[0][1][0:10])

for x in range(raw_number):
    lowers = [word.lower() for word in plot_data[0][x]]
    plot_data[0][x] = lowers

print(plot_data[0][1][0:10])

stop_words = set(stopwords.words('english'))

for x in range(raw_number):
    filtered_sentence = [w for w in plot_data[0][x] if not w in stop_words]
    plot_data[0][x] = filtered_sentence

print(plot_data[0][1][0:10])

snowball_stemmer = SnowballStemmer("english")
stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
print(stemmed_sentence[0:10])

porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
print(stemmed_sentence[0:10])

l = plot_data[0]
flatten = [item for sublist in l for item in sublist]
words = flatten
wordsunique = set(words)
wordsunique = list(wordsunique)

plottest = plot_data[0][0:1000]

worddic = {}
print("Starting word indexing:")
for i, doc in enumerate(plottest):
    print(str(i * 100 // len(plottest)) + "% completed")
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

np.save('worddic_1000.npy', worddic)
