from nltk.corpus import movie_reviews
from nltk.tag.stanford import StanfordTagger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from nltk.tag import StanfordPOSTagger
from Util import readBase, generateSimilarityMatrix, writeTxt
from Util import termFrequency
from Util import inverseDocumentFrequency
from CMUTweetTagger import runtagger_parse
import pickle
import numpy as np
import codecs
import sys

enable_perform_POS_tagging = True;
enable_perform_filtering_POS = True;

tweet = []
corpus_pre_processed = []
corpus_after_stop_word = []
corpus_after_stemming = []
corpus_after_POS = []
corpus_after_POS_filtering = []
corpus_before_preprocessing = []

## Text Pre-processing
tokensTwitter = []
tokensTwitterPOS = 0
emptyTweetPOS = 0
emptyTokensPOS = 0
emptyTweets = 0

def main():

    global tokensTwitterPOS
    global tokensTwitter
    global emptyTweets
    global emptyTweetPOS
    global emptyTokensPOS

    ## read corpus
    corpus_twitter = readBase('corpus/unbalanced_two_classes.csv')

    ## automated generate output ##
    #ordering = [(1,139), (2,139), (0,139)] #negative, neutral and positive balanced
    #ordering = [(1,177), (2,139), (0,182)] #negative, neutral and positive
    #ordering = [(1,139), (0,139)] #negative and positive balanced
    ordering = [(1,177), (0,182)] #negative and positive unbalanced


    output = []

    for i in ordering:
        output += [str(float(i[0])) + "\n"] * i[1]

    output[-1] = output[-1].replace("\n","") #remover ultima quebra de linha

    with codecs.open("results/unbalanced_two_classes_saida.txt", "w") as foutput:
        foutput.writelines(output)

    get_tweet = lambda twt:twt[0]

    corpus_twitter = map(get_tweet, corpus_twitter)

    if(enable_perform_POS_tagging):
    	corpus_after_POS = runtagger_parse(corpus_twitter, run_tagger_cmd="java -XX:ParallelGCThreads=2 -Xmx500m -jar ../ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar")

    corpus_before_preprocessing = get_only_tokens(corpus_after_POS) #tokenization eh feita no pos tagger

    corpus_after_POS = remove_tagger_probability(corpus_after_POS)

    count_tokens(corpus_after_POS)
    tokensTwitter = sum(tokensTwitter)

    if (enable_perform_filtering_POS):
        for tweet in corpus_after_POS:
            tok = []
            for j in tweet:
                if (j[1] is not None):
                    if (j[1] == 'A') or (j[1] == 'R') or (j[1] == 'V') or (j[1] == 'T') :
                        tok.append(j[0])
                    emptyTokensPOS += len(j[0]) == 0
            
            corpus_after_POS_filtering.append(tok)
            tokensTwitterPOS += len(tok)
            if len(tok) == 0: emptyTweetPOS += 1


    allTerms = []
    allTermsPOS = []

    allTerms = set(token for tweet in corpus_before_preprocessing for token in tweet)
    allTermsPOS = set(token for tweet in corpus_after_POS_filtering for token in tweet)

    tf = np.asarray([[termFrequency(w, z) for w in allTermsPOS] for z in corpus_after_POS_filtering])
    idf = np.asarray([[inverseDocumentFrequency(w , corpus_after_POS_filtering) for w in allTermsPOS]])
    matrix = [a * idf for a in tf]

    #writeTxt(matrix)

    final = generateSimilarityMatrix(matrix)
    np.savetxt('results/unbalanced_two_classes.txt',final)


    print('tokensTwitter:::::::::',tokensTwitter)
    print('tokensTwitterPOS::::::', tokensTwitterPOS)
    print('emptyTweetPOS:::::::::', emptyTweetPOS)
    print('emptyTokensPOS:::::::::', emptyTokensPOS)
    print('all terms ::::::::::::', len(allTerms))
    print('all terms afterPOS::::', len(allTermsPOS))

def remove_tagger_probability(corpus_tagged):
    iterator = lambda tupla:map(remove,tupla)
    remove = lambda tupla:tupla[:2]
    return map(iterator, corpus_tagged)

def count_tokens(corpus):
    global tokensTwitter
    get_tokens = lambda tweet:tokensTwitter.append(len(tweet))
    map(get_tokens, corpus)

def get_only_tokens(corpus):
	iterator = lambda tupla:map(remove,tupla)
	remove = lambda tupla:tupla[:1]
	return map(iterator, corpus)


if __name__ == '__main__':
    main()