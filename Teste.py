#-*- coding: utf-8 -*-

import os
from Util import readBase2, readLabels
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plotGraph(basePath = str, labelsPath = str, numSaida = int, name = str):

    base = readBase2(basePath)
    labels = readLabels(labelsPath)

    reduced_data = reduced_data = PCA(n_components=numSaida).fit_transform(np.array(base))

    # Geração do Gráficos
    c = {0: "r", 1: "g", 2: "b"}
    plt.figure(0)
    plt.title(name)
    for idx, label in enumerate(labels):
        plt.scatter(reduced_data[idx][0], reduced_data[idx][1], c=c[labels.tolist()[idx][0]])
    plt.savefig(os.path.join('plots', "sentment-140_balanced_specificPOS_three_classes.png"))


if __name__ == '__main__':
    basePath = 'Bracis/Database/sentiment-140/specific-POS-tagger/results/balanced_three_classes.txt'
    labelsPath = 'Bracis/Database/sentiment-140/specific-POS-tagger/results/balanced_three_classess_saida.txt'

    plotGraph(basePath, labelsPath, 3, 'sentment-140/balanced 3 classes/specificPos')