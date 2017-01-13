from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from Util import generateSimilarityMatrix, writeTxt
from Util import termFrequency
from Util import inverseDocumentFrequency
import pickle
import numpy as np
import codecs
from nltk import ngrams
import shutil, os
import sys
from random import shuffle

enable_perform_bigram = True;


def main():
    ## read corpus
    with open('../../../Bracis/Database/movie_reviews/corpus/balanced_two_classes.pkl', 'rb') as input:
        corpus_reviews = pickle.load(input)

    ngrams_configs = [(2, "word"), (3, "word"),(6,"char")]

    for ngram_config in ngrams_configs:

        path = "ngrams/" + str(ngram_config[0])+"gram_" + ngram_config[1]+"/"

        #config = ["balanced_two_classes", [(1,300),(0,300)]]
        config = ["unbalanced_two_classes", [(1,300),(0,100)]]

        # ## automated generate output ##
        # ordering = [(1,300), (0,300)] #respectly negative and positive
        # #ordering = [(1,300), (0,100)] #respectly negative and positive
        ordering = config[1]
        output = []
        for i in ordering:
            output += [str(float(i[0])) + "\n"] * i[1]

        output[-1] = output[-1].replace("\n","") #remover ultima quebra de linha

        with codecs.open(path + config[0] + "_saida.txt", "w") as foutput:
            foutput.writelines(output)
        ##

        corpus_after_POS = []
        corpus_after_POS_filtering = []
        corpus_before_preprocessing = []
        corpus_bigram_before_preprocessing = []
        corpus_bigram_after_pos = []

        # LABELS -> https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

        ## Text Pre-processing
        tokensReview = []
        tokensReviewPOS = 0
        emptyReviewPOS = 0
        emptyTokensPOS = 0

        #corpus_tagged = pos_tag_sents(corpus_reviews)

        corpus_before_preprocessing = [doc for doc in remove_label(corpus_reviews)]
        corpus_before_preprocessing = tuple(corpus_before_preprocessing)

        if (enable_perform_bigram):
            corpus_after_ngram = [get_ngrams(doc,ngram_config[0], ngram_config[1], tokenized=True) for doc in corpus_before_preprocessing]

        allTerms = []
        allTerms = set(bigram for review in corpus_after_ngram for bigram in review)

        print "Generating termFrequency"
        tf = np.asarray([[termFrequency(w, z) for w in allTerms] for z in corpus_after_ngram])
        print "Generating inverseDocumentFrequency"
        idf = np.asarray([[inverseDocumentFrequency(w, corpus_after_ngram) for w in allTerms]])
        print "Generating matrix"
        matrix = [a * idf for a in tf]

        #writeTxt(matrix)

        print "Generating similarity matrix"
        final = generateSimilarityMatrix(matrix)

        np.savetxt(path + config[0] + '.txt',final)

        informations = []
        informations.append('all terms ::::::::::::%d\n'%len(allTerms))

        with open(path+config[0] + "_result.txt",'w') as f:
            f.writelines(informations)

def remove_label(corpus):
    remove = lambda elemento:tuple(elemento[:1][0])
    return map(remove, corpus)

# def ngrams(sentence, n, tokenized = False):
#     if(not tokenized):
#         sentence = word_tokenize(sentence)
#     output = []
#     for i in range(len(sentence)-n+1):
#         g = ' '.join(ipt[i:i+n])
#         output.append(g)
#     return output

def get_ngrams(sentence, n, granularity, tokenized):
    if(tokenized):
        sentence = ' '.join(sentence)
    if(granularity == "word"):
        output_ngrams = ngrams(sentence.split(), n)
    else:
        output_ngrams = ngrams(sentence, n)
    output_ngrams = map(lambda element:' '.join(element), output_ngrams)
    return output_ngrams

if __name__ == '__main__':
    main();