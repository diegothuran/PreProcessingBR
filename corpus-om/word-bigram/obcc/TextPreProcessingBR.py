from nltk.tokenize import word_tokenize
from Util import generateSimilarityMatrix, writeTxt, readBase
from Util import termFrequency
from Util import inverseDocumentFrequency
import pickle
import numpy as np
import codecs
import sys
from random import shuffle
from nltk import ngrams
enable_perform_stemming = False
enable_perform_POS_tagging = True
enable_perform_filtering_POS = True
enable_perform_ngram = True

def main():
    ## read corpus
    corpus_twitter = readBase('../../../Bracis/Database/obcc/unigram-POS-tagger/corpus/unbalanced_two_class.csv')

    ngrams_configs = [(1, "word"), (2, "word"), (3, "word"), (6, "char")]

    for ngram_config in ngrams_configs:
        path = "ngrams/" + str(ngram_config[0]) + "gram_" + ngram_config[1] + "/"
        ## automated generate output ##
        #ordering = [(0,166), (1,1299)]
        #ordering = [(0,166), (1,1299), (2, 553)]
        #ordering = [(0,166), (1,166), (2, 166)]
        #ordering = [(0,166), (1,166)]l

        config = ["unbalanced_two_classes",[(0,166), (1,1299)]]
        #config = ["unbalanced_three_classes",[(0,166), (1,1299), (2, 553)] ]
        #config = ["balanced_three_classes",[(0,166), (1,166), (2, 166)]]
        #config = ["balanced_two_classes",[(0,166), (1,166)]]
        ordering = config[1]
        output = []
        for i in ordering:
            output += [str(float(i[0])) + "\n"] * i[1]

        output[-1] = output[-1].replace("\n","") #remover ultima quebra de linha

        with codecs.open(path + config[0] + "_saida.txt", "w") as foutput:
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

        corpus_before_preprocessing = [doc for doc in remove_label(corpus_twitter)]
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

        if (enable_perform_ngram):
            corpus_after_ngram = [get_ngrams(doc,ngram_config[0], ngram_config[1], tokenized = False) for doc in corpus_before_preprocessing]

        allTerms = []
        allTerms = set(token for tweet in corpus_after_ngram for token in tweet)

        print "Generating termFrequency"
        tf = np.asarray([[termFrequency(w, z) for w in allTerms] for z in corpus_after_ngram])
        print "Generating inverseDocumentFrequency"
        idf = np.asarray([[inverseDocumentFrequency(w, corpus_after_ngram) for w in allTerms]])
        print "Generating matrix"
        matrix = [a * idf for a in tf]

        #writeTxt(matrix)

        print "Generating similarity matrix"
        final = generateSimilarityMatrix(matrix)
        np.savetxt(path + config[0] + '.txt', final)

        informations = []
        informations.append('all terms ::::::::::::%d\n'%len(allTerms))

        with open(path+config[0] + "_result.txt",'w') as f:
            f.writelines(informations)


def remove_label(corpus):
    remove = lambda elemento:(elemento[0])
    return map(remove, corpus)

# def ngrams(ipt, n, tokenized = False):
#     if(not tokenized):
#         ipt = word_tokenize(ipt)
#     output = []
#     for i in range(len(ipt)-n+1):
#         g = ' '.join(ipt[i:i+n])
#         output.append(g)
#     return output
#
# def ngrams_dict(input, n, tokenized = False):
#     if(not tokenized):
#         input = word_tokenize(input)
#     output = {}
#     for i in range(len(input)-n+1):
#         g = ' '.join(input[i:i+n])
#         output.setdefault(g, 0)
#         output[g] += 1
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