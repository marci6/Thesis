import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

# CIFAR 100 - The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

def get(data_path,seed=0,pc_valid=0.10):
    data={}
    taskcla=[]
    size=[3,32,32]

    path = os.path.join(data_path, 'cifar10')
    if not os.path.isdir(path):
        os.makedirs(path)

    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    # CIFAR10
    dat={}
    dat['train']=datasets.CIFAR10(data_path,train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR10(data_path,train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
    data[0]={}
    data[0]['name']='cifar10'
    data[0]['ncla']=10
    data[0]['train']={'x': [],'y': []}
    data[0]['test']={'x': [],'y': []}
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            n=target.cpu().numpy()[0]
            data[0][s]['x'].append(image)
            data[0][s]['y'].append(n%2)

    # "Unify" and save
    for t in data.keys():
        for s in ['train','test']:
            data[t][s]['x']=torch.stack(data[t][s]['x']).view(-1,size[0],size[1],size[2])
            data[t][s]['y']=torch.LongTensor(np.array(data[t][s]['y'],dtype=int)).view(-1)
            torch.save(data[t][s]['x'], os.path.join(os.path.expanduser(path),'data'+str(t)+s+'x.bin'))
            torch.save(data[t][s]['y'], os.path.join(os.path.expanduser(path),'data'+str(t)+s+'y.bin'))


    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=seed),dtype=int)
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size
