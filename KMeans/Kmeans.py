#-*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from Util import readBase, readLabels, csv_writer, write_results
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
import sys
import time
import operator
import csv


if __name__=="__main__":

    resutados = []
    silhouetes =[]
    times = []
    accuracy = []
    MAX_EXECUTIONS = 25
    for iterations in range(MAX_EXECUTIONS):
        start = time.time()

        directory = "generalPosTagger/ngram-results/Execucao "+ str(iterations)
        if not os.path.exists(directory):
            os.makedirs(directory)

        orig_stdout = sys.stdout
        f = file(os.path.join(directory,'out'+str(iterations)+'.txt'), 'w')
        sys.stdout = f


        base = readBase('../corpus-om/word-bigram/obcc/ngrams/2gram_word/unbalanced_two_classes.txt') #matriz de similaridade entre os documentos, utilizando a distância entre os cosenos
        expected = readLabels('../corpus-om/word-bigram/obcc/ngrams/2gram_word/unbalanced_two_classes_saida.txt')
        reduced_data = PCA(n_components=2).fit_transform(np.array(base))
        km = KMeans(n_clusters=2, init='k-means++', max_iter=1000, n_init=1)
        km.fit(base)
        labels = km.labels_

        silhouetes.append( metrics.silhouette_score(expected, km.labels_))
        print("Silhouette Coefficient: %0.3f" %  metrics.silhouette_score(expected, km.labels_))
        print("Classification report for classifier %s:\n%s\n"
          % ("KMeans", metrics.classification_report(expected, labels)))

        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, labels))
        results = metrics.precision_recall_fscore_support(expected, labels)
        resutados.append(results)
        accuracy.append(metrics.accuracy_score(expected, labels, normalize=True))


        sys.stdout = orig_stdout
        f.close()

        # Geração do Gráficos
        c = {0:"r" , 1:"g", 2:"b"}
        plt.figure(0)
        for idx, label in enumerate(labels):
            plt.scatter(reduced_data[idx][0],reduced_data[idx][1], c=c[label])
        plt.savefig(os.path.join(directory, "figure.png"))

        c = {0: "r", 1: "g", 2: "b"}
        plt.figure(0)
        for idx in range(len(expected.tolist())):
            plt.scatter(reduced_data[idx][0], reduced_data[idx][1], c=c[expected.tolist()[idx][0]])
        plt.savefig(os.path.join(directory, "figure0.png"))

        times.append(time.time()-start)
    write_results(resutados,accuracy, times, silhouetes, "obcc-bigram-Kmeans_unbalanced_2_class_saida.csv")