from nltk.tokenize import word_tokenize
from Util import generateSimilarityMatrix, writeTxt, readBase
from Util import termFrequency
from Util import inverseDocumentFrequency
import random
import pickle
import numpy as np
import nltk
import codecs
import sys
from random import shuffle

enable_perform_stemming = False;
enable_perform_POS_tagging = True;
enable_perform_filtering_POS = True;
enable_perform_bigram = True;

def main():
    ## read corpus
    corpus_twitter = readBase('corpus/balanced_three_class.csv')
    stopwords = nltk.corpus.stopwords.words('portuguese')

    tweet_pre_process =[]
    i = 0
    while (i < len(corpus_twitter)):
        tweet_tokens = corpus_twitter[i][0].split()
        tweet_filtered = [w.lower() for w in tweet_tokens if not w in stopwords]
        content = (tweet_filtered, corpus_twitter[i][1])
        tweet_pre_process.append(content)
        i += 1

    ## automated generate output ##
    #ordering = [(0,166), (1,1299)] 
    #ordering = [(0,166), (1,1299), (2, 553)] 
    #ordering = [(0,166), (1,166), (2, 166)] 
    ordering = [(0,166), (1,166)]

    output = []
    for i in ordering:
        output += [str(float(i[0])) + "\n"] * i[1]

    output[-1] = output[-1].replace("\n","") #remover ultima quebra de linha

    with codecs.open("results/balanced_two_class_saida.txt", "w") as foutput:
        foutput.writelines(output)


    tweet = []
    corpus_pre_processed = []
    corpus_after_stop_word = []
    corpus_after_stemming = []
    corpus_after_POS = []
    corpus_after_POS_filtering = []
    corpus_before_preprocessing = []
    corpus_bigram_before_preprocessing = []
    corpus_bigram_after_pos = []


    ## Text Pre-processing
    tokensTwitter = []
    tokensTwitterPOS = 0
    emptyTwitterPOS = 0
    emptyTokensPOS = 0
    nullTags = 0
    emptyTags = 0

    corpus_before_preprocessing = [doc for doc in remove_label(tweet_pre_process)]
    corpus_before_preprocessing = tuple(corpus_before_preprocessing)
    # x = [token for tweet in corpus_bigram_before_preprocessing for token in tweet]
    # y = []
    # for token in x:
    #     if (not token in y):
    #         y.append(token)
    #     else:
    #         print token

    map(lambda tweet:tokensTwitter.append(len(tweet)), corpus_before_preprocessing)
    tokensTwitter = sum(tokensTwitter)

    if (enable_perform_bigram):
        corpus_after_bigram = [ngrams(doc,2, tokenized = False) for doc in corpus_before_preprocessing]

    allTerms = []
    allTerms = set(token for tweet in corpus_after_bigram for token in tweet)

    print "Generating termFrequency"
    tf = np.asarray([[termFrequency(w, z) for w in allTerms] for z in corpus_after_bigram])
    print "Generating inverseDocumentFrequency"
    idf = np.asarray([[inverseDocumentFrequency(w, corpus_after_bigram) for w in allTerms]])
    print "Generating matrix"
    matrix = [a * idf for a in tf]

    #writeTxt(matrix)

    print "Generating similarity matrix"
    final = generateSimilarityMatrix(matrix)

    np.savetxt('results/balanced_two_class.txt',final)

    print('all terms ::::::::::::', len(allTerms))

def remove_label(corpus):
    remove = lambda elemento:(elemento[0])
    return map(remove, corpus)


def ngrams(ipt, n, tokenized = False):
    if(not tokenized):
        ipt = word_tokenize(ipt)
    output = []
    for i in range(len(ipt)-n+1):
        g = ' '.join(ipt[i:i+n])
        output.append(g)
    return output


def ngrams_dict(input, n, tokenized = False):
    if(not tokenized):
        input = word_tokenize(input)
    output = {}
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    return output


if __name__ == '__main__':
    main();