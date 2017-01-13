#-*- coding: utf-8 -*-
__author__ = 'diego'
import math
import random
import numpy as np
from sklearn.metrics import silhouette_score
from Util import readBase, readLabels, write_results
from sklearn.decomposition import PCA
import time
from sklearn import metrics
import os, sys
from random import shuffle


class ArtificialBeeColony(object):
    base = readBase('../corpus-om/word-bigram/sentiment-140/results/unbalanced_two_classes.txt')
    labels = readLabels('../corpus-om/word-bigram/sentiment-140/results/unbalanced_two_classes_saida.txt')

    c = list(zip(base, labels))

    random.shuffle(c)

    base, labels = zip(*c)
    base = np.array(base)
    labels = np.array(labels)

    np = 20
    food_number = int(np/2)
    limit = 100
    max_cycle = 50

    d = len(base)
    lower_bound = 0
    upper_bound = 1

    epochs = 25

    foods = [[0] * d] * food_number
    f = [0] * food_number
    fitness = [0] * food_number
    trial = [0] * food_number
    prob = [0] * food_number
    solution = [0] * d

    obj_val_sol = None
    fitness_sol = None

    neighbor = None
    param_to_change = None
    global_min = None
    global_params = [None] * d
    global_mins = [None] * epochs

    r = None

    @staticmethod
    def calculate_fitness(fun):
        if fun >= 0:
            return 1/(fun + 1)
        else:
            return 1 + abs(fun)

    def generate_random_number(self):
        self.r = int(random.random() * 32767) / (float(32767) + float(1))

    def memorize_best_food_source(self):
        for i in range(self.food_number):
            if self.f[i] < self.global_min:
                self.global_min = self.f[i]
                for j in range(self.d):
                    self.global_params[j] = self.foods[i][j]

    def init(self, index):
        for j in range(self.d):
            self.generate_random_number()
            self.foods[index][j] = random.randint(self.lower_bound, self.upper_bound)
            self.solution[j] = self.foods[index][j]
        self.f[index] = self.calculate_function(self.solution)
        self.fitness[index] = self.calculate_fitness(self.f[index])
        self.trial[index] = 0

    def initial(self):
        for i in range(self.food_number):
            self.init(i)
        self.global_min = self.f[0]
        for i in range(self.d):
            self.global_params[i] = self.foods[0][i]

    def send_employed_bees(self):
        for i in range(self.food_number):
            self.generate_random_number()
            self.param_to_change = int(self.r * self.d)
            self.generate_random_number()
            self.neighbor = int(self.r * self.food_number)
            for j in range(self.d):
                self.solution[j] = self.foods[i][j]
            self.generate_random_number()
            self.solution[self.param_to_change] = self.foods[i][self.param_to_change] + (self.foods[i][self.param_to_change] - self.foods[self.neighbor][self.param_to_change]) * (self.r - 0.5) * 2
            if self.solution[self.param_to_change] < self.lower_bound:
                self.solution[self.param_to_change] = self.lower_bound
            if self.solution[self.param_to_change] > self.upper_bound:
                self.solution[self.param_to_change] = self.upper_bound

            # self.obj_val_sol = self.calculate_function(self.solution)
            # self.fitness_sol = self.calculate_fitness(self.obj_val_sol)
            self.fitness_sol = self.sillhouete(self.solution, self.base)

            if self.fitness_sol > self.fitness[i]:
                self.trial[i] = 0
                for j in range(self.d):
                    self.foods[i][j] = self.solution[j]
                    self.f[i] = self.obj_val_sol
                    self.fitness[i] = self.fitness_sol
            else:
                self.trial[i] += 1

    def calculate_probabilities(self):
        maxfit = self.fitness[0]
        for i in range(1, self.food_number):
            if self.fitness[i] > maxfit:
                maxfit = self.fitness[i]
        for i in range(self.food_number):
            if maxfit == 0:
                maxfit = 0.1
            self.prob[i] = (0.9 * (self.fitness[i]/maxfit)) + 0.1

    def send_onlooker_bees(self):
        i = 0
        t = 0
        while t < self.food_number:
            self.generate_random_number()
            if self.r < self.prob[i]:
                t += 1
                self.generate_random_number()
                self.param_to_change = int(self.r * self.d)
                self.generate_random_number()
                self.neighbor = int(self.r * self.food_number)
                while self.neighbor == i:
                    self.generate_random_number()
                    self.neighbor = int(self.r * self.food_number)
                for j in range(self.d):
                    self.solution[j] = self.foods[i][j]
                self.generate_random_number()
                self.solution[self.param_to_change] = self.foods[i][self.param_to_change] + (self.foods[i][self.param_to_change] - self.foods[self.neighbor][self.param_to_change]) * (self.r - 0.5) * 2
                if self.solution[self.param_to_change] < self.lower_bound:
                    self.solution[self.param_to_change] = self.lower_bound
                if self.solution[self.param_to_change] > self.upper_bound:
                    self.solution[self.param_to_change] = self.upper_bound
                #self.obj_val_sol = self.calculate_function(self.solution)
                #self.fitness_sol = self.calculate_fitness(self.obj_val_sol)
                self.fitness_sol = self.sillhouete(self.solution, self.base)

                if self.fitness_sol > self.fitness[i]:
                    self.trial[i] = 0
                    for j in range(self.d):
                        self.foods[i][j] = self.solution[j]
                    self.f[i] = self.obj_val_sol
                    self.fitness[i] = self.fitness_sol
                else:
                    self.trial[i] += 1
                i += 1
                if i == self.food_number:
                    i = 0

    def send_scout_bees(self):
        maxtrial_index = 0
        for i in range(1, self.food_number):
            if self.trial[i] > self.trial[maxtrial_index]:
                maxtrial_index = i
        if self.trial[maxtrial_index] >= self.limit:
            self.init(maxtrial_index)

    def calculate_function(self, sol):
        return self.sphere(sol)

    def sphere(self, sol):
        top = 0
        for j in range(self.d):
            top += sol[j]**2
        return top

    def sillhouete(self, sol= [], base = []):
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
        cluster_labels = np.array(sol)
        silhouette_avg = silhouette_score(temp, cluster_labels)

        return silhouette_avg


abc = ArtificialBeeColony()
j = 0
mean = 0
times = []
resultados = []
accuracy = []
gs = []
for run in range(abc.epochs):
    initial = time.time()
    directory = "Execucao " + str(run)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Redirecionamento da sída do Console para o txt.
    orig_stdout = sys.stdout
    f = file(os.path.join(directory, 'out' + str(run + 1) + '.txt'), 'w')
    sys.stdout = f
    reduced_data = PCA(n_components=2).fit_transform(np.array(abc.base))


    abc.initial()
    abc.memorize_best_food_source()



    for iter in range(abc.max_cycle):
        abc.send_employed_bees()
        abc.calculate_probabilities()
        abc.send_onlooker_bees()
        abc.memorize_best_food_source()
        abc.send_scout_bees()

    duracao = time.time() - initial
    times.append(duracao)
    # cálculo das métricas
    resultados.append(metrics.precision_recall_fscore_support(abc.labels, abc.solution))
    accuracy.append(metrics.accuracy_score(abc.labels, abc.solution, normalize=True))
    # print das métricas F-score e etc
    print("Classification report for classifier %s:\n%s\n"
          % ("CLUDIPSO", metrics.classification_report(abc.labels, abc.solution)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(abc.labels, abc.solution))
    gs.append(metrics.silhouette_score(abc.labels, np.array(abc.solution)))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(abc.labels, np.array(abc.solution)))

    sys.stdout = orig_stdout
    f.close()
write_results(resultados, accuracy, times, gs, 'sentiment-bigram-unbalanced-2-classes.csv')


