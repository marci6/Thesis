# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 12:28:46 2021

@author: MARCELLOCHIESA
"""

import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms

########################################################################################################################

def get(data_path,seed,fixed_order=False,pc_valid=0):
    data={}
    taskcla=[]
    size=[3,32,32]

    # MNIST
    mean=(0.4514,)
    std=(0.1993,)
    dat={}
    dat['train']=datasets.SVHN(data_path,split='train',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.SVHN(data_path,split='test',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data[0]={}
    data[0]['name']='SVHN'
    data[0]['ncla']=10

    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        data[0][s]={'x': [],'y': []}
        for image,target in loader:
            label=target.numpy()-1
            data[0][s]['x'].append(image)
            data[0][s]['y'].append(label)

    # "Unify" and save
    for n in range(1):
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'].clone()
        data[t]['valid']['y']=data[t]['train']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################
