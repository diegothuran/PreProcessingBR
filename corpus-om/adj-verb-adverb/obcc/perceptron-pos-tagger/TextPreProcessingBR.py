from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from Util import generateSimilarityMatrix, writeTxt, readBase
from Util import termFrequency
from Util import inverseDocumentFrequency
import pickle
import numpy as np
import codecs
import sys
from random import shuffle

enable_perform_POS_tagging = True;
enable_perform_filtering_POS = True;
enable_perform_smothing = False;

def main():
    ## read corpus
    corpus_twitter = readBase('corpus/balanced_three_class.csv')

    ## automated generate output ##
    #ordering = [(0,166), (1,1299)] 
    #ordering = [(0,166), (1,1299), (2, 553)] 
    ordering = [(0,166), (1,166), (2, 166)] 
    #ordering = [(0,166), (1,166)] 

    output = []
    for i in ordering:
        output += [str(float(i[0])) + "\n"] * i[1]

    output[-1] = output[-1].replace("\n","") #remover ultima quebra de linha

    with codecs.open("results/balanced_three_class_saida.txt", "w") as foutput:
        foutput.writelines(output)
    #

    tweet = []
    corpus_pre_processed = []
    corpus_after_stop_word = []
    corpus_after_stemming = []
    corpus_after_POS = []
    corpus_after_POS_filtering = []
    corpus_before_preprocessing = []

    # LABELS -> https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    tagger = pickle.load(open("PerceptronTagger.pickle"))

    ## Text Pre-processing
    tokensTwitter = []
    tokensTwitterPOS = 0
    emptyTwitterPOS = 0
    emptyTokensPOS = 0
    nullTags = 0
    emptyTags = 0

    corpus_before_preprocessing = [word_tokenize(doc) for doc in remove_label(corpus_twitter)]

    map(lambda tweet:tokensTwitter.append(len(tweet)), corpus_before_preprocessing)
    tokensTwitter = sum(tokensTwitter)

    if (enable_perform_POS_tagging):
        corpus_after_POS = tagger.tag_sents(corpus_before_preprocessing)
        print "Pos Tagging finished..."

    if (enable_perform_filtering_POS):
        for tweet in corpus_after_POS:
            tok = []
            for j in tweet:
                if (j[1] is not None):
                    if ((j[1] == 'ADJ') or (j[1] == 'ADV') or (j[1] == 'ADV-KS') or (j[1] == 'ADV-KS-REL') or (j[1] == 'V') or (j[1] == 'VAUX')):
                        tok.append(j[0])
                else:
                    nullTags += 1
                emptyTokensPOS += int(len(j[0]) == 0)
                emptyTags += int(len(j[1]) == 0)
            corpus_after_POS_filtering.append(tok)
            tokensTwitterPOS += len(tok)
            if len(tok) == 0: emptyTwitterPOS += 1

    allTerms = []
    allTermsPOS = []

    allTerms = set(token for tweet in corpus_before_preprocessing for token in tweet)
    allTermsPOS = set(token for tweet in corpus_after_POS_filtering for token in tweet)

    print "Generating termFrequency"
    tf = np.asarray([[termFrequency(w, z) for w in allTermsPOS] for z in corpus_after_POS_filtering])
    print "Generating inverseDocumentFrequency"
    idf = np.asarray([[inverseDocumentFrequency(w , corpus_after_POS_filtering) for w in allTermsPOS]])
    print "Generating matrix"
    matrix = [a * idf for a in tf]

    #writeTxt(matrix)

    print "Generating similarity matrix"
    final = generateSimilarityMatrix(matrix)

    np.savetxt('results/balanced_three_class.txt',final)

    print('tokensTwitter:::::::::',tokensTwitter)
    print('tokensTwitterPOS::::::', tokensTwitterPOS)
    print('emptyTweetsAfterPOS:::::::::', emptyTwitterPOS)
    print('emptyTokensAfterPOS:::::::::', emptyTokensPOS)
    # print ('emptyTags', emptyTags)
    # print ('nullTags', nullTags)
    print('all terms ::::::::::::', len(allTerms))
    print('all terms afterPOS::::', len(allTermsPOS))

def remove_label(corpus):
    remove = lambda elemento:elemento[:1][0]
    return map(remove, corpus)

if __name__ == '__main__':
    main();