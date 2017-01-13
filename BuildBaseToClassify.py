#-*- coding: utf-8 -*-

from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import word_tokenize
from Util import readBase, generateSimilarityMatrix
from Util import termFrequency
from Util import inverseDocumentFrequency
from os import mkdir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

i=0
j=0
k=0
l=0

#### Leitura do Corpus ####
#corpus_om = readBase('dataset/balanced_corpus/corpus_three_class_balanced/tweets-total-csv-3-class-balanced.csv', header=1)
#corpus_om = readBase('dataset/tweets-total-csv-4-class-test.csv', header=0)
corpus_om, saidas = readBase('dataset/two_class_test.csv', header=1)
saidas = np.array(saidas)

#### Criação da StopList ####
pt_stop_words = set(stopwords.words('portuguese'))
pt_stop_words.update('@', '#')

tweet_pre_process =  []
tweet_term_freq = []

#### Stopword Removal ####
while (i<len(corpus_om)):
    # tweet_tokens = word_tokenize(corpus_om[i][0])
    tweet_tokens = corpus_om[i][0].split()
    tweet_filtered = [w.lower() for w in tweet_tokens if not w in pt_stop_words]
    content = (tweet_filtered, corpus_om[i][1])
    tweet_pre_process.append(content)
    i += 1

#### Stemming ####

allDocuments = []
allTerms = []

while (k<len(tweet_pre_process)):
    allDocuments.append(tweet_pre_process[k][0])
    k += 1

#print("*** all documents ***", allDocuments)

while (j<len(tweet_pre_process)):
    #term = [w for w in tweet_pre_process[j][0]]
    temp = ([w for w in tweet_pre_process[j][0] if not w in allTerms])
    for z in range(len(temp)):
        allTerms.append(temp[z])
    j += 1


tf = np.asarray([[termFrequency(w, z) for w in allTerms] for z in allDocuments])
idf = np.asarray([[inverseDocumentFrequency(w , allDocuments) for w in allTerms]])
matrix = [a * idf for a in tf]
final = generateSimilarityMatrix(matrix)

np.savetxt('basesExtraidas/two_class.txt',final)
np.savetxt('basesExtraidas/two_class_saidas.txt',saidas)
