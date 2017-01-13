import csv
import sys

from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import word_tokenize
from Util import readBase, generateSimilarityMatrix
from Util import termFrequency
from Util import inverseDocumentFrequency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

i=0
j=0
k=0
l=0

#corpus_om = readBase('dataset/balanced_corpus/corpus_three_class_balanced/tweets-total-csv-3-class-balanced.csv', header=0)
corpus_om = readBase('dataset/two_class_test.csv', header=1)
pt_stop_words = set(stopwords.words('portuguese'))

tweet_pre_process =  []
tweet_term_freq = []

while (i<len(corpus_om)):
    # tweet_tokens = word_tokenize(corpus_om[i][0])
    tweet_tokens = corpus_om[i][0].split()
    tweet_filtered = [w.lower() for w in tweet_tokens if not w in pt_stop_words]
    content = (tweet_filtered, corpus_om[i][1])
    tweet_pre_process.append(content)
# print(" *******Documnent ID ", i+1," *******")
# print("Tweet", corpus_om[i][0])
# print("Teewt tokens", tweet_tokens)
# print("Teewt stopwords", tweet_filtered)
# print("Tweet pre-processed", tweet_pre_process[i])
    i += 1

# print("len tweet_pre_process ", len(tweet_pre_process))
# print("len tweet_pre_process linha 1 ", len(tweet_pre_process[0][0]))
# print("len tweet_pre_process linha 1 ", len(tweet_pre_process[1][0]))
# print("len tweet_pre_process linha 1 ", len(tweet_pre_process[2][0]))
# print("len tweet_pre_process linha 1 ", len(tweet_pre_process[3][0]))

matrix = []
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


while (l<len(allDocuments)):
    temp = []
    for w in allTerms:
        tf = termFrequency(w,allDocuments[l])
        idf = inverseDocumentFrequency(w,allDocuments)
        content = (l+1, w ,tf*idf)
        temp.append(tf*idf)
    matrix.append(temp)
    l += 1
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(allDocuments)
# print(tfidf_matrix)

#print(matrix)
#print type(matrix)
final = generateSimilarityMatrix(matrix)
np.savetxt('base_teste.txt', final)