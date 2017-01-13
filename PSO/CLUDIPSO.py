#-*- coding: utf-8 -*-
import sys, csv
__author__ = 'diego'
from Util import *
import collections
from pprint import pprint
from random import random, randint
from LinearDecay import LinearDecay
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
import os, operator


#vetores de saída para serem salvos
bests = []
times = []
gs = []
resultados = []
MAX_EXECUTIONS = 25

for iterations in range(MAX_EXECUTIONS):
    #redirecionamento do output
    directory = "Execucao "+ str(iterations)
    if not os.path.exists(directory):
        os.makedirs(directory)

    orig_stdout = sys.stdout
    f = file(os.path.join(directory,'out'+str(iterations)+'.txt'), 'w')
    sys.stdout = f

    base = readBase('../basesExtraidas/adj-adv-noum-verb/balanced_three_class.txt') #matriz de similaridade entre os documentos, utilizando a distância entre os cosenos
    expected = readLabels('../basesExtraidas/adj-adv-noum-verb/balanced_three_class_saida.txt')
    # variaveis globais
    pop_size = 20  #tamanho do enxame = número de partículas
    dimensions = len(expected)  #quantidade de dimensoes em cada particula = número de documentos a serem clusterizados
    # para K = 3 clusters, uma dimensão pode assumir um valor no intervalo [0-2], representado pelas variáveis upperBounds e bottonBounds
    upperBounds = 1 #maior valor que uma dimensao pode tomar
    bottonBounds = 0  #menor valor que uma dimensao pode tomar

    maxIteration = 50 #numero de iteracoes do enxame
    upperVelocity = 5  #velocidade maxima permitida pelo enxame
    bottonVelocity = -5  #velocidade minima permitida pelo enxame

    # varivaeis de busca do enxame
    w = 0.9
    c1 = 1
    c2 = 1

    class Particle:
        pass

    reduced_data = PCA(n_components=2).fit_transform(np.array(base))
    particles = [] #vetor de particulas
    start = time.time() #aqui o algoritmo comeca a funcionar

    #inicializacao do vetor de partículas
    for i in range(pop_size):
        p = Particle()
        p.positions = [randint(bottonBounds, upperBounds) for j in range(dimensions)]
        p.fitness = -1

        p.velocities = [randint(bottonBounds, upperBounds) for j in range(dimensions)] #####????? Não seria upperVelocity e bottonVelocity
        particles.append(p)

    #inicializacao do gbets
    gbest = particles[0]

    #atualizacao do enxame
    i= 0
    fitnessExecution = []
    while i < maxIteration :
        for p in particles:
            r1 = random()
            r2 = random()
            same = 0 #####?????
            fitness, error = fitnessSilhouete(p.positions, base)
            fitnessExecution.append(fitness)

            if fitness > p.fitness:
                p.fitness = fitness
                p.best = p.positions

            if fitness > gbest.fitness:

                gbest = p

            #####????? atualiza posicao e velociade Se velocidade > randoNum
            for j in range(len(p.positions)):
                if(p.velocities[j]/upperBounds > random()):
                    p.velocities[j] =tryUpdateVelocity(LinearDecay(w,0.4,maxIteration,False).apply(j) * p.velocities[j] + c1 * r1 * (p.best[j] - p.positions[j]) \
                            + c2 * r2 * (gbest.positions[j] - p.positions[j]), upperVelocity, bottonVelocity)
                    p.positions[j] = tryUpdatePosition(p.velocities[j] + p.positions[j], upperBounds, bottonBounds)
            p.positions = applyMutation(p.positions)
            same += 1
        i  += 1
    end = time.time()

    print '\nParticle Swarm Optimisation\n'
    print 'PARAMETERS\n','-'*9
    print 'Population size : ', pop_size
    print 'Dimensions      : ', dimensions
    print 'c1              : ', c1
    print 'c2              : ', c2
    print 'function        :  silhouete'

    print 'RESULTS\n', '-'*7
    print 'gbest fitness   : ', gbest.fitness
    #print 'gbest params    : ', gbest.positions
    print 'iterations      : ', i
    print 'time duration   :', end - start, 's'


    clustering = collections.defaultdict(list)

    for idx, label in enumerate(gbest.positions):
            clustering[label].append(idx)

    print "Classificacao Final:"
    pprint(dict(clustering))

    #cálculo das métricas
    resultados.append(metrics.precision_recall_fscore_support(expected, gbest.positions))

#    gbests.append(gbest.fitness)
#    times.append(end - start)


    #print das métricas F-score e etc
    print("Classification report for classifier %s:\n%s\n"
          % ("CLUDIPSO", metrics.classification_report(expected, gbest.positions)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, gbest.positions))
    gs.append(metrics.silhouette_score(expected, np.array(gbest.positions)))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(expected, np.array(gbest.positions)))

    c = {0:"r" , 1:"g", 2:"b"}
    #fechamento do arquivo de resultados
    sys.stdout = orig_stdout
    f.close()
    plt.figure(1)
    plt.plot(fitnessExecution)
    plt.axis([0, maxIteration, 0.4, 1.6])
    plt.savefig(os.path.join(directory,"figure "+str(iterations) +".png"))
    plt.figure(2)
    for idx, label in enumerate(gbest.positions):
        plt.scatter(reduced_data[idx][0],reduced_data[idx][1], c=c[label])
    plt.savefig(os.path.join(directory,"figure2 "+str(iterations) +".png"))
csv_writer2(gs,"gs.csv")
media =  np.array(reduce(operator.add, np.mean(np.array(resultados), axis=2)))/MAX_EXECUTIONS
csv_writer2(media,"resultado_final.csv")
