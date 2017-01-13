#-*- coding: utf-8 -*-
__author__ = 'diego'
from random import uniform, randint
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import csv


import math
import numpy as np
import operator



#funcao de fitness de teste
def fitnessFunc(chromosome):
        """F6 Griewank's function
        multimodal, symmetric, inseparable"""
        part1 = 0

        for i in range(len(chromosome)):
            part1 += chromosome[i]**2
        part2 = 1
        for i in range(len(chromosome)):
            part2 *= math.cos(float(chromosome[i]) / math.sqrt(i+1))
        return 1 + (float(part1)/4000.0) - float(part2)

#funcao de fitness usando a silhoueta
#ela esta sempre retornando o mesmo valor, acho que nao apliquei muito bem #####?????
def fitnessSilhouete(positions = [], base = []):
    temp = []
    for i in positions:
        temp.append(base[i])
    temp = np.array(temp)
    cluster_labels = np.array(positions)
    silhouette_avg = silhouette_score(temp, cluster_labels)
    error = 1- silhouette_avg
    return silhouette_avg, error

#funcao de mutacao sem a equacao do pm, mas esta trocando de posicao dois valores aleatorios de uma particula
def applyMutation(positions):
    rand1 = randint(0, (len(positions)-1))
    rand2 = rand1
    while rand1 == rand2:
        rand2 = randint(0, (len(positions)-1))

    temp = positions[rand1]
    positions[rand1] = positions[rand2]
    positions[rand2] = temp
    return positions

def tryUpdatePosition(value, upperBounds, bottonBounds):
    if(value> upperBounds):
        return int(upperBounds)
    elif (value < bottonBounds):
        return int(bottonBounds)
    else:
        return int(value)

def tryUpdateVelocity(value, upperVelocity, bottonVelocity):
    if(value> upperVelocity):
        return int(upperVelocity)
    elif (value < bottonVelocity):
        return int(bottonVelocity)
    else:
        return int(value)

def generateBase(size):
    base = []
    for i in range(10):
        temp = [uniform(-1,1) for j in range(size)]
        base.append(temp)

    for i in range(len(base)):
        for j in range(len(base[0])):
            if i == j:
                base[i][j] = 1

    np.savetxt('base.txt', base)

def readBase(endereco = str):
    with open(endereco,'r') as ins:
        base = []
        for line in ins:
            temp = []
            for element in line.split(' '):
                temp.append(float(element))
            base.append(temp)

    return base

def compareClusters(predicter = [], labels= []):
    pass

def readLabels(endereco):
    with open(endereco,'r') as ins:
        saida = []
        for line in ins:
            temp = []
            for element in line.split(' '):
                temp.append(float(element))
            saida.append(temp)
    return np.array(saida)

def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Precision", "Recall", "Fscore", "Support"])
        writer.writerow(data)
        #writer.writerow("Media: "+ np.mean(data))

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