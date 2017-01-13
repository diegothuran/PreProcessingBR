from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from Util import generateSimilarityMatrix, writeTxt
from Util import termFrequency
from Util import inverseDocumentFrequency
from nltk import pos_tag_sents
import pickle
import numpy as np
import codecs
import sys
from random import shuffle

enable_perform_stemming = False;
enable_perform_POS_tagging = True;
enable_perform_filtering_POS = True;
enable_perform_smothing = False;

def main():
    ## read corpus
    with open('corpus/balanced_two_classes.pkl', 'rb') as input:
        corpus_reviews = pickle.load(input)

    ## automated generate output ##
    ordering = [(1,300), (0,300)] #respectly negative and positive 

    output = []
    for i in ordering:
        output += [str(float(i[0])) + "\n"] * i[1]

    output[-1] = output[-1].replace("\n","") #remover ultima quebra de linha

    with codecs.open("results/balanced_two_classes_saida.txt", "w") as foutput:
        foutput.writelines(output)
    ##

    review = []
    corpus_pre_processed = []
    corpus_after_stop_word = []
    corpus_after_stemming = []
    corpus_after_POS = []
    corpus_after_POS_filtering = []
    corpus_before_preprocessing = []

    # LABELS -> https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    ## Text Pre-processing
    tokensReview = []
    tokensReviewPOS = 0
    emptyReviewPOS = 0
    emptyTokensPOS = 0

    #corpus_tagged = pos_tag_sents(corpus_reviews)

    corpus_before_preprocessing = remove_label(corpus_reviews)

    map(lambda tweet:tokensReview.append(len(tweet)), corpus_before_preprocessing)
    tokensReview = sum(tokensReview)

    if (enable_perform_POS_tagging):
        corpus_after_POS = pos_tag_sents(corpus_before_preprocessing)
        print "Pos Tagging finished..."

    if (enable_perform_filtering_POS):
        for review in corpus_after_POS:
            tok = []
            for j in review:
                if (j[1] is not None):
                    if (j[1] == 'RB') or (j[1] == 'RBR') or (j[1] == 'RBS') or (j[1] == 'JJ') or (j[1] == 'JJR') or (j[1] == 'JJS') or (j[1] == 'VB') or (j[1] == 'VBD') or (j[1] == 'VBG') or (j[1] == 'VBN') or (j[1] == 'VBP') or (j[1] == 'VBZ'):
                        tok.append(j[0])
                    emptyTokensPOS += len(j[0]) == 0
            corpus_after_POS_filtering.append(tok)
            tokensReviewPOS += len(tok)
            if len(tok) == 0: emptyReviewPOS += 1
        

    allTerms = []
    allTermsPOS = []

    allTerms = set(token for review in corpus_before_preprocessing for token in review)
    allTermsPOS = set(token for review in corpus_after_POS_filtering for token in review)

    print "Generating termFrequency"
    tf = np.asarray([[termFrequency(w, z) for w in allTermsPOS] for z in corpus_after_POS_filtering])
    print "Generating inverseDocumentFrequency"
    idf = np.asarray([[inverseDocumentFrequency(w , corpus_after_POS_filtering) for w in allTermsPOS]])
    print "Generating matrix"
    matrix = [a * idf for a in tf]

    #writeTxt(matrix)

    print "Generating similarity matrix"
    final = generateSimilarityMatrix(matrix)

    np.savetxt('results/balanced_two_classes.txt',final)


    print('tokensReview:::::::::',tokensReview)
    print('tokensReviewPOS::::::', tokensReviewPOS)
    print('emptyReviewPOS:::::::::', emptyReviewPOS)
    print('emptyTokensPOS:::::::::', emptyTokensPOS)
    print('all terms ::::::::::::', len(allTerms))
    print('all terms afterPOS::::', len(allTermsPOS))


def remove_label(corpus):
    remove = lambda elemento:elemento[:1][0]
    return map(remove, corpus)


if __name__ == '__main__':
    main();