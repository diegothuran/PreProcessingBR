#-*- coding: utf-8 -*-
__author__ = 'diego'
'''
    Arquivo com algumas funções de Utilidade que auxiliam na execução do PSO
'''


from random import  randint
from sklearn.metrics import silhouette_score
import csv
import operator
import numpy as np


def readLabels(endereco):
    '''
        Função responsável por ler as labels do base de dados
    :param endereco: local em que o arquivo de labels está armazenado
    :return: vetor com todos os labels
    '''
    with open(endereco,'r') as ins:
        saida = []
        for line in ins:
            temp = []
            for element in line.split(' '):
                temp.append(float(element))
            saida.append(temp)
    return np.array(saida).astype(int)


def readBase(endereco = str):
    '''
        Função responsável por ler o arquivo com a base de dados extraída
    :param endereco: local em que o arquivo da base está armazenado
    :return: vetor com a base de dados
    '''
    with open(endereco,'r') as ins:
        base = []
        for line in ins:
            temp = []
            for element in line.split(' '):
                temp.append(float(element))
            base.append(temp)

    return base


def fitnessSilhouete(positions = [], base = []):
    '''
    funcao de fitness usando a função silhoueta para avaliar a melhoria do enxame
    :param positions: label de cada uma das partículas
    :param base: valor de cada uma das partículas
    :return: o valor da função silhoeta
    '''
    temp = []
    for i in range(len(base)):
        temp.append(base[i])
    temp = np.array(temp)
    cluster_labels = np.array(positions)
    silhouette_avg = silhouette_score(temp, cluster_labels)

    return silhouette_avg

def write_results(resultados =[], accuracy = [], times =[], silhoetes =[], path= str):

    '''
        Função responsável por escrever um arquivo csv com todos os resultados obtidos
    :param resultados: resultados de cada iteração do PSO
    :param times: o tempo de execução para cada iteração do PSO
    :param silhoetes: Avaliação da semelhança dos clusters antes e depois da execução do PSO
            em cada execução
    :param path: caminho e nome do arquivo a ser salvo
    '''


    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([" "," ", "Classe 0", "Classe 1","Classe 2",  "All Class"])
        writer.writerow(["Execution Number", "Accuracy ","Precision/Recall/F-Score",
                         "Precision/Recall/F-Score","Precision/Recall/F-Score",
                         "Precision/Recall/F-Score/Support",
                         "Silhouette Coefficient","Time (seconds)"])

        print_vetor = [[z+1, ac, a[0], a[1], a[2],np.mean(a, axis=1), b, c]for z,ac, a, b, c in zip(range(0,len(resultados)),accuracy, resultados,silhoetes,times)]

        for i in range(len(print_vetor)):
            writer.writerow(print_vetor[i])

        media = np.array(reduce(operator.add, np.mean(np.array(resultados), axis=2)))/len(resultados)
        writer.writerow([" ","Accuracy","Precison", "Recall", "F-Score"])
        writer.writerow(["Media Final:",np.mean(accuracy),media[0], media[1], media[2]])