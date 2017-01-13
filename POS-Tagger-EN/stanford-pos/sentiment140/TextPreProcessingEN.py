from nltk.corpus import movie_reviews
from nltk.tag.stanford import StanfordTagger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from nltk.tag import StanfordPOSTagger
from Util import readBase, generateSimilarityMatrix, writeTxt
from Util import termFrequency
from Util import inverseDocumentFrequency
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
    ## read corpus
    corpus_twitter = readBase('corpus/testdata.csv')

    ## automated generate output ##
    ordering = {0:177, 2:139, 4:182}

    output = []

    for i in ordering.items():
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

    ## POS tagger
    tagger = StanfordPOSTagger('../stanford-postagger/models/english-bidirectional-distsim.tagger',"../stanford-postagger/stanford-postagger.jar", java_options='-Xmx3800m')
    # LABELS -> https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    ## Text Pre-processing
    tokensTwitter = 0
    tokensTwitterPOS = 0
    emptyTweetPOS = 0
    emptyTokensPOS = 0

    for document in corpus_twitter:
        tokens = word_tokenize(document[0])
        tokensTwitter += len(tokens)
        tweet = [w.lower() for w in tokens]
        
        corpus_before_preprocessing.append(tweet)
        
        if (enable_remove_stopwords):
            tweet = [w.lower() for w in tokens if not w in pt_br_stop_words]
            corpus_after_stop_word.append(tweet)

        if (enable_perform_stemming):
            tweet = [stemmerPT.stem(w) for w in tweet]
            corpus_after_stemming.append(tweet)

        if (enable_perform_POS_tagging):
            tweet = tagger.tag(tweet)
            corpus_after_POS.append(tweet)


        if (enable_perform_filtering_POS):
            tok = []
            for j in tweet:
                if (j[1] is not None):
                    if (j[1] == 'RB') or (j[1] == 'RBR') or (j[1] == 'RBS') or (j[1] == 'NN') or (j[1] == 'NNS') or (j[1] == 'NNP') or (j[1] == 'NNPS') or (j[1] == 'JJ') or (j[1] == 'JJR') or (j[1] == 'JJS') or (j[1] == 'VB') or (j[1] == 'VBD') or (j[1] == 'VBG') or (j[1] == 'VBN') or (j[1] == 'VBP') or (j[1] == 'VBZ'):
                        tok.append(j[0])
                    emptyTokensPOS += len(j[0]) == 0
            corpus_after_POS_filtering.append(tok)
            tokensTwitterPOS += len(tok)
            if len(tok) == 0: emptyTweetPOS += 1

        content = (tweet, document[1])
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

    #np.savetxt('balanced_two_class_vsm',matrix)

    final = generateSimilarityMatrix(matrix)
    np.savetxt('unbalanced_three_class.txt',final)

    print('tokensTwitter:::::::::', tokensTwitter)
    print('tokensTwitterPOS::::::', tokensTwitterPOS)
    print('emptyTweetPOS:::::::::', emptyTweetPOS)
    print('emptyTokensPOS:::::::::', emptyTokensPOS)
    print('all terms ::::::::::::', len(allTerms))
    print('all terms afterPOS::::', len(allTermsPOS))



if __name__ == '__main__':
    main()