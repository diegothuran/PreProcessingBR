from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from Util import readBase, generateSimilarityMatrix
from Util import termFrequency
from Util import inverseDocumentFrequency
import cPickle as pickle
import numpy as np

enable_remove_stopwords = False
enable_remove_usernames = False
enable_remove_emojis = False
enable_remove_urls = False
enable_remove_tweetstyle = False
enable_remove_punctuation = False
enable_perform_stemming = False
enable_perform_POS_tagging = True
enable_perform_filtering_POS = True
enable_perform_smothing = False

## read corpus
corpus_twitter = readBase('../dataset/unbalanced_corpus/unbalanced_two_class.csv', header=1)

tweet = []
corpus_pre_processed = []
corpus_after_stop_word = []
corpus_after_stemming = []
corpus_after_POS = []
corpus_after_POS_filtering = []
corpus_before_preprocessing = []

## build stop word list for brazilian portuguese
pt_br_stop_words = set(stopwords.words('portuguese'))
pt_br_stop_words.update('@', '#')

## Stemmer for Portuguese
stemmerPT = RSLPStemmer()

## POS tagger
tagger = pickle.load(open("taggerUnigramPT.pickle"))

## Text Pre-processing
i = 0
tokensTwitter = 0
tokensTwitterPOS = 0
emptyTweetPOS = 0
#while (i<len(corpus_twitter)):
while (i<10):
    tokens = word_tokenize(corpus_twitter[i][0])
    tokensTwitter += len(tokens)
    # tokens = corpus_twitter[i][0].split()    tokens = word_tokenize(corpus_twitter[i][0])
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
                # if ( j[1].find('adv') <> -1) or ( j[1].find('adj') <> -1)  or ( j[1].find('n') <> -1 ):
                if (j[1] == 'adv') or (j[1] == 'n') or (j[1] == 'adj') or (j[1] == 'n-adj') or (j[1] == 'v-fin') or (
                    j[1] == 'v-inf') or (j[1] == 'v-pcp') or (j[1] == 'v-ger'):
                    tok.append(j[0])
        corpus_after_POS_filtering.append(tok)
        tokensTwitterPOS += len(tok)
        if len(tok) == 0: emptyTweetPOS += 1

    content = (tweet, corpus_twitter[i][1])
    corpus_pre_processed.append(content)
#    print(corpus_twitter[i][0])
#    print(content)
#    print(corpus_after_POS_filtering)
    i += 1


allTerms = []

l=0
while (l<len(corpus_before_preprocessing)):
    temp = ([w for w in corpus_before_preprocessing[l] if not w in allTerms])
    for z in range(len(temp)):
        allTerms.append(temp[z])
    l += 1

## Similarity Matrix
allTermsPOS = []

j=0
while (j<len(corpus_after_POS_filtering)):
    temp = ([w for w in corpus_after_POS_filtering[j] if not w in allTermsPOS])
    for z in range(len(temp)):
        allTermsPOS.append(temp[z])
    j += 1


tf = np.asarray([[termFrequency(w, z) for w in allTermsPOS] for z in corpus_after_POS_filtering])
idf = np.asarray([[inverseDocumentFrequency(w , corpus_after_POS_filtering) for w in allTermsPOS]])
matrix = [a * idf for a in tf]

np.savetxt('matrix.txt',matrix)

#np.savetxt('balanced_two_class_vsm',matrix)

final = generateSimilarityMatrix(matrix)
#np.savetxt('unbalanced_three_class',final)



print('tokensTwitter:::::::::',tokensTwitter)
print('tokensTwitterPOS::::::', tokensTwitterPOS)
print('emptyTweetPOS:::::::::', emptyTweetPOS)
print('all terms ::::::::::::', len(allTerms))
print('all terms afterPOS::::', len(allTermsPOS))


# print('corpus_twitter_len::::::::::',len(corpus_twitter))
#
# for elem in corpus_twitter:
#     print('corpus_twitter::::::::::',elem)
#
# for elem in corpus_pre_processed:
#     print('corpus_pre_processed::::::::::',elem)
#
# for elem in corpus_after_POS_filtering:
#     print('corpus_after_POS_filtering::::::::::',elem)
