# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:43:39 2021

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
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()
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
        x = self.quant(x)
        x = torch.nn.functional.relu(self.fc1(x, sample))
        y=[]
        if self.head == 'multi':
            for t,i in self.taskcla:
                xx = self.classifier[t](x, sample)
                xx = torch.nn.functional.log_softmax(xx, dim=1)
                y.append(self.dequant(xx))
            return y
        elif self.head == 'single':
            y.append(self.classifier(x, sample))
            return [torch.nn.functional.log_softmax(yy, dim=1) for yy in y]


def Net(args):
    return BayesianMLP(args)
