from nltk.corpus import movie_reviews
from nltk.tag.stanford import StanfordTagger
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
from Util import readBase, generateSimilarityMatrix, writeTxt
from Util import termFrequency
from Util import inverseDocumentFrequency
import pickle
import numpy as np
import codecs
import sys

enable_perform_stemming = False;
enable_perform_POS_tagging = True;
enable_perform_filtering_POS = True;
enable_perform_smothing = False;

def main():
    ## read corpus
    corpus_reviews = [(list(movie_reviews.words(fileid)), category)
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)]

    ## automated generate output ##
    ordering = {'1':1000, '0':1000} #respectly negative and positive 

    output = []
    for i in ordering.items():
        output += [i[0] + "\n"] * i[1]

    output[-1] = output[-1].replace("\n","") #remover ultima quebra de linha

    with codecs.open("balanced_two_class_saida.txt", "w") as foutput:
        foutput.writelines(output)
    ##

    review = []
    corpus_pre_processed = []
    corpus_after_stop_word = []
    corpus_after_stemming = []
    corpus_after_POS = []
    corpus_after_POS_filtering = []
    corpus_before_preprocessing = []

    ## POS tagger
    tagger = StanfordPOSTagger('../stanford-postagger/models/english-bidirectional-distsim.tagger',"../stanford-postagger/stanford-postagger.jar", java_options='-Xmx3024m')
    # LABELS -> https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    ## Text Pre-processing
    tokensReview = 0
    tokensReviewPOS = 0
    emptyReviewPOS = 0
    emptyTokensPOS = 0

    for document in corpus_reviews:
        review = document[0]
        tokensReview += len(review)
        corpus_before_preprocessing.append(review)

        if (enable_perform_POS_tagging):
            #print corpus_reviews.index(document)
            review = tagger.tag(review)
            corpus_after_POS.append(review)

        if (enable_perform_filtering_POS):
            tok = []
            for j in review:
                if (j[1] is not None):
                    if (j[1] == 'RB') or (j[1] == 'RBR') or (j[1] == 'RBS') or (j[1] == 'NN') or (j[1] == 'NNS') or (j[1] == 'NNP') or (j[1] == 'NNPS') or (j[1] == 'JJ') or (j[1] == 'JJR') or (j[1] == 'JJS') or (j[1] == 'VB') or (j[1] == 'VBD') or (j[1] == 'VBG') or (j[1] == 'VBN') or (j[1] == 'VBP') or (j[1] == 'VBZ'):
                        tok.append(j[0])
                    emptyTokensPOS += len(j[0]) == 0
            corpus_after_POS_filtering.append(tok)
            tokensReviewPOS += len(tok)
            if len(tok) == 0: emptyReviewPOS += 1

        content = (review, document[1])
        corpus_pre_processed.append(content)


    allTerms = []

    for twt in corpus_before_preprocessing:
        temp = ([w for w in twt if not w in allTerms])
        for i in temp:
            allTerms.append(i)

    ## Similarity Matrix
    allTermsPOS = []

    for twt in corpus_after_POS_filtering:
        temp = ([w for w in twt if not w in allTermsPOS])
        for i in temp:
            allTermsPOS.append(i)


    tf = np.asarray([[termFrequency(w, z) for w in allTermsPOS] for z in corpus_after_POS_filtering])
    idf = np.asarray([[inverseDocumentFrequency(w , corpus_after_POS_filtering) for w in allTermsPOS]])
    matrix = [a * idf for a in tf]

    #writeTxt(matrix)

    final = generateSimilarityMatrix(matrix)
    np.savetxt('balanced_two_class.txt',final)

    print('tokensReview:::::::::',tokensReview)
    print('tokensReviewPOS::::::', tokensReviewPOS)
    print('emptyReviewPOS:::::::::', emptyReviewPOS)
    print('emptyTokensPOS:::::::::', emptyTokensPOS)
    print('all terms ::::::::::::', len(allTerms))
    print('all terms afterPOS::::', len(allTermsPOS))

if __name__ == '__main__':
    main();