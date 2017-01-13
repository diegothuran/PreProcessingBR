#-*- coding: utf-8 -*-
__author__ = 'diego'
'''
    Classe que define uma partícula do PSO
'''

from random import randint, random
from Util import *

class Particle:

    def __init__(self, bottonBounds = int, upperBounds = int, upperVelocity = int, bottonVelocity = int, dimensions = int,
                 fitness = -1):

        '''
            Construtor da classe Partícula que recebe e inicializa os atributos

            :param bottonBounds - Limite inferior até onde a partícula pode navegar no espaço de busca
            :param upperBounds - Limite superior até onde a partícula pode navegar no espaço de busca
            :param upperVelocity - velociade máxima permitida à partícula
            :param bottonVelocity - velociade mínima permitida à partícula
            :param dimensions - número de dimensões em cada partícula
            :param fitness - fitness inicial de cada partícula
        '''

        self.bottonBounds = bottonBounds
        self.upperBounds = upperBounds
        self.upperVelocity = upperVelocity
        self.bottonVelocity = bottonVelocity
        self.dimensions = dimensions
        self.positions = [randint(self.bottonBounds, self.upperBounds) for j in range(self.dimensions)]
        self.fitness = fitness
        self.velocities = [randint(self.bottonBounds, self.upperBounds) for j in range(self.dimensions)]
        self.best = self.positions


