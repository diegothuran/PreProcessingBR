#-*- coding: utf-8 -*-

from sklearn.cluster import AgglomerativeClustering
from Util import readBase, readLabels, write_results
from sklearn import metrics
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import random
import os

resultados = []
gs = []
times = []
accuracy = []
NUM_EXECUTIONS = 25
for i in range(NUM_EXECUTIONS):
    base = readBase('../corpus-om/word-bigram/sentiment-140/results/unbalanced_two_classes.txt')
    labels = readLabels('../corpus-om/word-bigram/sentiment-140/results/unbalanced_two_classes_saida.txt')
    c = list(zip(base, labels))

    random.shuffle(c)

    base, labels = zip(*c)
    base = np.array(base)
    labels = np.array(labels)
    reduced_data = PCA(n_components=3).fit_transform(np.array(base))
    clustering = None

    #print lan
    start = time.time()
    for linkage in ('ward', 'average', 'complete'):

        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=2)
        clustering.fit(base)
    times.append(time.time() - start)
    c = metrics.confusion_matrix(clustering.labels_, labels)

    resultados.append(metrics.precision_recall_fscore_support(clustering.labels_, labels))
    gs.append(metrics.silhouette_score(labels, clustering.labels_))
    accuracy.append(metrics.accuracy_score(clustering.labels_, labels, normalize=True))

if len(resultados)>0:
    write_results(resultados=resultados, times=times,accuracy=accuracy, silhoetes=gs, path="Agglomerative_sentiment_bigram_unbalanced_2_lasses.csv")
    c = {0: "r", 1: "g", 2: "b"}

    # gráfico dos dados depois da clusterização
    plt.figure(2)
    plt.title("Data after clustering")
    for idx, label in enumerate(clustering.labels_):
        plt.scatter(reduced_data[idx][0], reduced_data[idx][1], c=c[label])
    plt.savefig("figure2.png")

    # gráfico dos dados antes da clusterização
    plt.figure(3)
    plt.title("Data before clustering")
    for idx in range(len(labels)):
        plt.scatter(reduced_data[idx][0], reduced_data[idx][1], c=c[labels.tolist()[idx][0]])
    plt.savefig("figure0.png")