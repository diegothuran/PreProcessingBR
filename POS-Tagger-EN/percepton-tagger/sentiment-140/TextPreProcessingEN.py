from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from Util import readBase, generateSimilarityMatrix, writeTxt
from Util import termFrequency
from Util import inverseDocumentFrequency
from nltk import pos_tag_sents
from collections import OrderedDict
import pickle
import numpy as np
import codecs
import sys

enable_remove_stopwords = False;
enable_remove_usernames = False;
enable_remove_emojis = False;
enable_remove_urls = False;
enable_remove_tweetstyle = False;
enable_remove_punctuation = False;
enable_perform_stemming = False;
enable_perform_POS_tagging = True;
enable_perform_filtering_POS = True;
enable_perform_smothing = False;

def main():
    ## read corpus ##
    corpus_twitter = readBase('corpus/testdata.csv')

    ## automated generate output ##
    ordering = [(1,177), (2,139), (0,182)] #negative, neutral and positive

    output = []

    for i in ordering:
        output += [str(float(i[0])) + "\n"] * i[1]

    output[-1] = output[-1].replace("\n","") #remover ultima quebra de linha

    with codecs.open("unbalanced_three_class_saida.txt", "w") as foutput:
        foutput.writelines(output)
    ##

    tweet = []
    corpus_pre_processed = []
    corpus_after_stop_word = []
    corpus_after_stemming = []
    corpus_after_POS = []
    corpus_after_POS_filtering = []
    corpus_before_preprocessing = []

    # LABELS -> https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    ## Text Pre-processing
    tokensTweet = []
    tokensTweetPOS = 0
    emptyTweetPOS = 0
    emptyTokensPOS = 0

    #corpus_tagged = pos_tag_sents(corpus_tweets)

    corpus_before_preprocessing = [word_tokenize(doc) for doc in remove_label(corpus_twitter)]

    map(lambda tweet:tokensTweet.append(len(tweet)), corpus_before_preprocessing)
    tokensTweet = sum(tokensTweet)

    if (enable_perform_POS_tagging):
        corpus_after_POS = pos_tag_sents(corpus_before_preprocessing)

    if (enable_perform_filtering_POS):
        for tweet in corpus_after_POS:
            tok = []
            for j in tweet:
                if (j[1] is not None):
                    if (j[1] == 'RB') or (j[1] == 'RBR') or (j[1] == 'RBS') or (j[1] == 'NN') or (j[1] == 'NNS') or (j[1] == 'NNP') or (j[1] == 'NNPS') or (j[1] == 'JJ') or (j[1] == 'JJR') or (j[1] == 'JJS') or (j[1] == 'VB') or (j[1] == 'VBD') or (j[1] == 'VBG') or (j[1] == 'VBN') or (j[1] == 'VBP') or (j[1] == 'VBZ'):
                        tok.append(j[0])
                    emptyTokensPOS += len(j[0]) == 0
            corpus_after_POS_filtering.append(tok)
            tokensTweetPOS += len(tok)
            if len(tok) == 0: emptyTweetPOS += 1

    allTerms = []
    allTermsPOS = []

    allTerms = set(token for tweet in corpus_before_preprocessing for token in tweet)
    allTermsPOS = set(token for tweet in corpus_after_POS_filtering for token in tweet)

    tf = np.asarray([[termFrequency(w, z) for w in allTermsPOS] for z in corpus_after_POS_filtering])
    idf = np.asarray([[inverseDocumentFrequency(w , corpus_after_POS_filtering) for w in allTermsPOS]])
    matrix = [a * idf for a in tf]

    writeTxt(matrix)

    final = generateSimilarityMatrix(matrix)
    np.savetxt('balanced_two_class.txt',final)

    print('tokensTweet:::::::::',tokensTweet)
    print('tokensTweetPOS::::::', tokensTweetPOS)
    print('emptyTweetPOS:::::::::', emptyTweetPOS)
    print('emptyTokensPOS:::::::::', emptyTokensPOS)
    print('all terms ::::::::::::', len(allTerms))
    print('all terms afterPOS::::', len(allTermsPOS))


def remove_label(corpus):
    remove = lambda elemento:elemento[:1][0]
    return map(remove, corpus)


if __name__ == '__main__':
    main();