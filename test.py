import pickle
from nltk.corpus import stopwords
import numpy as np
from Util import generateSimilarityMatrix
from Util import termFrequency
from Util import inverseDocumentFrequency

base = pickle.load(open('BRACIS-corpus/movieReviews/balanced_two_classes.pkl', 'rb'))

reviews = []
outputs = []
dictionary = {u'pos':1, u'neg':2, "Irrelevante" :0, "Neutro":0}

for review in base:
    reviews.append(review[0])
    outputs.append(dictionary[review[1]])

en_stop_words = set(stopwords.words('english'))
i =0

tweet_pre_process = []
#### Stopword Removal ####
while (i<len(reviews)):
    # tweet_tokens = word_tokenize(corpus_om[i][0])
    tweet_tokens = reviews[i]
    tweet_filtered = [w.lower() for w in tweet_tokens if not w in en_stop_words]
    content = (tweet_filtered, reviews[i][1])
    tweet_pre_process.append(content)
    i += 1

#### Stemming ####

allDocuments = []
allTerms = []
i = 0
while (i<len(tweet_pre_process)):
    allDocuments.append(tweet_pre_process[i][0])
    i += 1

#print("*** all documents ***", allDocuments)
i = 0
while (i<len(tweet_pre_process)):
    #term = [w for w in tweet_pre_process[j][0]]
    temp = ([w for w in tweet_pre_process[i][0] if not w in allTerms])
    for z in range(len(temp)):
        allTerms.append(temp[z])
    i += 1

tf = np.asarray([[termFrequency(w, z) for w in allTerms] for z in allDocuments])
idf = np.asarray([[inverseDocumentFrequency(w , allDocuments) for w in allTerms]])
matrix = [a * idf for a in tf]
final = generateSimilarityMatrix(matrix)



np.savetxt('basesExtraidas/two_class.txt',final)
np.savetxt('basesExtraidas/two_class_saidas.txt',outputs)
