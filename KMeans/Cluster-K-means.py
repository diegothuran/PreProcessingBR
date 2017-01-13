import string
import collections

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from nltk.stem import RSLPStemmer
from Util import readBase
from sklearn import metrics
import numpy as np

def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    text = text.translate(string.punctuation)
    tokens = word_tokenize(text)

    if stem:
        stemmerPT = RSLPStemmer()
        tokens = [stemmerPT.stem(t) for t in tokens]

    return tokens


def cluster_texts(texts, clusters=3,stopwords=[],processing_tp=''):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=processing_tp,
                                 stop_words=stopwords,
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)

    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)

    clustering = collections.defaultdict(list)

    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
    return clustering




if __name__ == "__main__":
    base = readBase('../basesExtraidas/adj-adv-noum-verb/two_class.txt')
    matrix_sim = np.array(base)
    km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
    km.fit(matrix_sim)
    labels = km.labels_

    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(matrix_sim, km.labels_, sample_size=1000))


'''
    texts = []
    corpus_om = readBase('../dataset/balanced_corpus/corpus_three_class_balanced/tweets-total-csv-3-class-balanced.csv', header=0)
    for i in range(len(corpus_om)):
        texts.append(corpus_om[i][0])
    pt_stop_words = set(stopwords.words('portuguese'))
    pt_stop_words.update('@','#')
    clusters = cluster_texts(texts, 3, pt_stop_words,process_text)
    pprint(dict(clusters))
'''