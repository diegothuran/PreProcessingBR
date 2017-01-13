import csv, math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys

#locale.setlocale(locale.LC_CTYPE,'pt_br')

def readBase(csvFile = str, header = 0):
    base = []
    with open(csvFile, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='\"')
        for row in spamreader:
            label = row[0]
            tweet =  unicode(row[14], encoding='utf-8', errors='ignore')
            base.append(tuple([tweet.lower(), label]))
        return base

# Normalized Term Frequency
def termFrequency(term, document):
    normalizeDocument = document
    if len(normalizeDocument) > 0: return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))
    else: return 0

# Inverse Term Frequency
def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    for doc in allDocuments:
        if term.lower() in allDocuments:
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1

    if numDocumentsWithThisTerm > 0:
        return 1.0 + math.log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0

def generateSimilarityMatrix(data= []):
    #matrix1 = [cosine_similarity(a,b) for a, b in zip(data, data)]
    matrix = []
    for i in range(len(data)):
        temp = []
        for j in range(len(data)):
                temp.append(cosine_similarity(data[i],data[j]))
        matrix.append(temp)
    return matrix

def writeTxt(data= []):
    mat = []
    for line in data:
        mat.append(line[0])
    np.savetxt('matrix.txt', mat)
