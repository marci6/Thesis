# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:45:48 2021

@author: MARCELLOCHIESA
"""
import torch
import numpy as np
from .FC import BayesianLinear

class BayesianMLP(torch.nn.Module):
    
    def __init__(self, args):
        super(BayesianMLP, self).__init__()

        ncha,size,_= args.inputsize
        self.taskcla= args.taskcla
        self.samples = args.samples
        self.device = args.device
        self.sbatch = args.sbatch
        self.init_lr = args.lr
        self.head = args.head
        # dim=60  #100k
        # dim=1200
        dim=args.nhid
        nlayers=args.nlayers

        self.fc1 = BayesianLinear(ncha*size*size, dim, args)
        if nlayers==2:
            self.fc2 = BayesianLinear(dim, dim, args)
            
        self.classifier = torch.nn.ModuleList()
        if self.head == 'multi':
            for t,n in self.taskcla:
                self.classifier.append(BayesianLinear(dim, n, args))
        elif self.head == 'single':
            self.classifier.append(BayesianLinear(dim, self.taskcla[0][1]*len(self.taskcla), args))


    def prune(self,mask_modules):
        for module, mask in mask_modules.items():
            module.prune_module(mask)


    def forward(self, x, sample=False):
        x = x.view(x.size(0),-1)
        x = torch.nn.functional.relu(self.fc1(x, sample))
        y=[]
        if self.head == 'multi':
            for t,i in self.taskcla:
                y.append(self.classifier[t](x, sample))
            return [torch.nn.functional.log_softmax(yy, dim=1) for yy in y]
        elif self.head == 'single':
            y.append(self.classifier(x, sample))
            return [torch.nn.functional.log_softmax(yy, dim=1) for yy in y]


def Net(args):
    return BayesianMLP(args)

