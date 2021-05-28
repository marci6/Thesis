# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:11:20 2021

@author: MARCELLOCHIESA
"""

class Parameters:
    def __init__(self):
        self.device = 'cuda:0'
        self.sbatch = 64
        self.lr = 0.01
        self.nlayers = 1
        self.nhid = 1200
        self.samples = 10
        self.rho = -3
        self.std1 = 0.0
        self.std2 = 6.0
        self.pi = 0.25
        self.seed = 0
        self.experiment = 'mnist5'
        self.approach = 'ucb'
        self.inputsize = None
        self.taskcla = None
        self.data_path = '../data/'
        self.nepochs = 200
        self.resume = 'no'
        self.arch = 'mlp'
        self.num_tasks = 1