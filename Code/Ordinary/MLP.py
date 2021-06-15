# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:16:28 2021

@author: MARCELLOCHIESA
"""

import torch
from torch import nn

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, args):
        super().__init__()
        ncha,size,_= args.inputsize
        self.taskcla= args.taskcla
        self.samples = args.samples
        self.device = args.device
        self.sbatch = args.sbatch
        self.init_lr = args.lr
        self.qat = args.qat
        # dim=60  #100k
        # dim=1200
        dim=args.nhid
        self.nlayers=args.nlayers
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()
        self.fc1 = nn.Linear(ncha*size*size, 2*dim)
        if self.nlayers==2:
            self.fc2 = nn.Linear(2*dim, 2*dim)
    
        self.classifier = torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.classifier.append(nn.Linear(2*dim, n))


  def forward(self, x):
    '''Forward pass'''
    if self.qat:
        x = self.quant(x)
    x = x.view(x.size(0),-1)
    x = torch.nn.functional.relu(self.fc1(x))
    if self.nlayers==2:
        x = torch.nn.functional.relu(self.fc2(x))
    y=[]
    if self.qat:
        for t,i in self.taskcla:
            xx = self.classifier[t](x)
            y.append(self.dequant(xx))
        return [torch.nn.functional.log_softmax(yy, dim=1) for yy in y]
    else:
        for t,i in self.taskcla:
            y.append(self.classifier[t](x))
        return [torch.nn.functional.log_softmax(yy, dim=1) for yy in y]

def Net(args):
    return MLP(args)