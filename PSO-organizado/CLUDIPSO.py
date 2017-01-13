#-*- coding: utf-8 -*-
__author__ = 'diego'

from Particle import Particle
from Util import *
from random import random
from LinearDecay import LinearDecay
import collections
from pprint import pprint
import os, sys, time
import matplotlib.pyplot as plt
from sklearn import metrics
import operator
from sklearn.decomposition import PCA


class CLUDIPSO:
    '''
        Classe que define um PSO do tipo CLUDIPSO

    '''

    def __init__(self, popSize, bottonBounds = int, upperBounds = int, upperVelocity = int,
                 bottonVelocity = int, dimensions = int, w=0.9,c1 = 1, c2 = 1):

        '''
            Construtor da classe CLUDIPSO

            :param popSize - Tamanho do enxame
            :param bottonBounds - Limite inferior até onde a partícula pode navegar no espaço de busca
            :param upperBounds - Limite superior até onde a partícula pode navegar no espaço de busca
            :param upperVelocity - velociade máxima permitida à partícula
            :param bottonVelocity - velociade mínima permitida à partícula
            :param dimensions - número de dimensões em cada partícula
            :param w - Variável que controla o quanto a partícula pode navegar
            :param c1 - Variável que controla a variação entre explotaition e exploration
            :param c2 - Variável que controla a variação entre explotaition e exploration
        '''

        self.popSize = popSize
        self.bottonBounds = bottonBounds
        self.upperBounds = upperBounds
        self.upperVelocity = upperVelocity
        self.bottonVelocity = bottonVelocity
        self.dimensions = dimensions
        self.population = [Particle(self.bottonBounds, self.upperBounds, self.upperVelocity,
                                    self.bottonVelocity, self.dimensions) for a in range(self.popSize)]

        self.gbest = self.population[0]
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def run(self, numbersOfInterations, base):
        '''
            Método que executa o CLUDIPSO
            :param numbersOfInterations: número de iterações pelas quais o PSO vai ser executado
            :param base: base de dados para que a função de fitness obtenha resultado de ftnesse a cada iteração
            :return fitnessExecution: variável com todos os fitness de cada iteração
        '''
        fitnessExecution =[]

        for i in range(numbersOfInterations):
            for p in self.population:
                r1 = random()
                r2 = random()
                fitness = fitnessSilhouete(p.positions, base)
                fitnessExecution.append(fitness)

                if fitness > p.fitness:
                    p.fitness = fitness
                    p.best = p.positions

                if fitness > self.gbest.fitness:
                    self.gbest = p

                for j in range(len(p.positions)):
                    if (p.velocities[j] / p.upperBounds > random()):
                        p.velocities[j] = tryUpdateVelocity(LinearDecay(self.w, 0.4, numbersOfInterations, False).apply(j)
                                                            * p.velocities[j] + self.c1 * r1 * (p.best[j] - p.positions[j]) \
                            + self.c2 * r2 * (self.gbest.positions[j] - p.positions[j]), p.upperVelocity, p.bottonVelocity)
                        p.positions[j] = tryUpdatePosition(p.velocities[j] + p.positions[j], p.upperBounds, p.bottonBounds)
                p.positions = applyMutation(p.positions)
            fitnessExecution.append(self.gbest.fitness)
        return fitnessExecution

    def prinResults(self, duracao):

        '''

        :param duracao: variável que armazena o tempo de execuçaõ de uma iteração do CLUDIPSO
        :return:
        '''

        print '\nParticle Swarm Optimisation\n'
        print 'PARAMETERS\n', '-' * 9
        print 'Population size : ', self.popSize
        print 'Dimensions      : ', self.dimensions
        print 'c1              : ', self.c1
        print 'c2              : ', self.c2
        print 'function        :  silhouete'

        print 'RESULTS\n', '-' * 7
        print 'gbest fitness   : ', self.gbest.fitness
        print 'time duration   :', duracao, 's'

        clustering = collections.defaultdict(list)

        for idx, label in enumerate(self.gbest.positions):
            clustering[label].append(idx)

        print "Classificacao Final:"
        pprint(dict(clustering))




if __name__ == '__main__':
    start = time.time()
    resultados = []
    gs = []
    times = []
    accuracy = []
    numberOfExecutions = 25
    maxIterations = 50
    for i in range(numberOfExecutions):
        directory = "Execucao " + str(i)
        if not os.path.exists(directory):
            os.makedirs(directory)
        #Redirecionamento da sída do Console para o txt.
        orig_stdout = sys.stdout
        f = file(os.path.join(directory, 'out' + str(i) + '.txt'), 'w')
        sys.stdout = f

        #Leitura da base de dados
        base = readBase(
            '../corpus-om/word-bigram-acento/obcc/results/balanced_two_class.txt')  # matriz de similaridade entre os documentos, utilizando a distância entre os cosenos
        expected = readLabels('../corpus-om/word-bigram-acento/obcc/results/balanced_two_class_saida.txt')

        #Redução da base de dados para plotagem
        reduced_data = PCA(n_components=2).fit_transform(np.array(base))

        #iniciação do enxame com:
        #   popSize = 20 partículas
        #   bottonBouns = 0, upperBounds = 1 for two classes
        #   bottonBouns = 0, upperBounds = 2 for three classes
        #   bottonVelocity = -5, upperVelocity = 5
        #   dimension = número de documentos
        PSO = CLUDIPSO(popSize=20, bottonBounds=0, upperBounds=1, upperVelocity=5, bottonVelocity = -5, dimensions=len(expected))

        #execução do PSO
        fitnessExecution = PSO.run(maxIterations, base)

        #print dos resultados do PSO
        duracao = time.time() - start
        PSO.prinResults(duracao)
        times.append(duracao)

        # cálculo das métricas
        resultados.append(metrics.precision_recall_fscore_support(expected, PSO.gbest.positions))
        accuracy.append(metrics.accuracy_score(expected, PSO.gbest.positions,normalize = True))

        # print das métricas F-score e etc
        print("Classification report for classifier %s:\n%s\n"
              % ("CLUDIPSO", metrics.classification_report(expected, PSO.gbest.positions)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, PSO.gbest.positions))
        gs.append(metrics.silhouette_score(expected, np.array(PSO.gbest.positions)))
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(expected, np.array(PSO.gbest.positions)))

        c = {0: "r", 1: "g", 2: "b"}
        # fechamento do arquivo de resultados
        sys.stdout = orig_stdout
        f.close()

        #plotagem das figuras
            #gráfico de convergêndia
        plt.figure(1)
        plt.title("Convergence")
        plt.plot(fitnessExecution)
        plt.axis([0, maxIterations, 0.4, 1.6])
        plt.savefig(os.path.join(directory, "figure " + str(i) + ".png"))

        #gráfico dos dados depois da clusterização
        plt.figure(2)
        plt.title("Data after clustering")
        for idx, label in enumerate(PSO.gbest.positions):
            plt.scatter(reduced_data[idx][0], reduced_data[idx][1], c=c[label])
        plt.savefig(os.path.join(directory, "figure2 " + str(i) + ".png"))

        #gráfico dos dados antes da clusterização
        plt.figure(3)
        plt.title("Data before clustering")
        for idx in range(len(expected.tolist())):
            plt.scatter(reduced_data[idx][0], reduced_data[idx][1], c=c[expected.tolist()[idx][0]])
        plt.savefig(os.path.join(directory, "figure0 " + str(i) + ".png"))

    write_results(resultados,accuracy, times,gs,"dpsomut-acento-word-bigram-obcc-results-balanced_two_classes.csv")
