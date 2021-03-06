# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:04:00 2021

@author: MARCELLOCHIESA
"""
import numpy as np
import torch
from torchvision import datasets,transforms

########################################################################################################################

# Omniglot divided into 50 sets.

def get(data_path,seed,fixed_order=False,pc_valid=0):
    data={}
    taskcla=[]
    size=[1,105,105]
    counter_samples = 0 
    # Omniglot
    mean=(0.9179,)
    std=(0.2745,)
    dat={}
    dat['background']=datasets.Omniglot(data_path,background=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['eval']=datasets.Omniglot(data_path,background=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    # 10 tasks
    data[0]={}
    data[0]['name']='mnist-0-4'
    data[0]['ncla']=5
    data[1]={}
    data[1]['name']='mnist-5-9'
    data[1]['ncla']=5
    data[2]={}
    data[2]['name']='mnist-10-14'
    data[2]['ncla']=5
    data[3]={}
    data[3]['name']='mnist-15-19'
    data[3]['ncla']=5
    data[4]={}
    data[4]['name']='mnist-20-24'
    data[4]['ncla']=5
    data[5]={}
    data[5]['name']='mnist-25-29'
    data[5]['ncla']=5
    data[6]={}
    data[6]['name']='mnist-30-34'
    data[6]['ncla']=5
    data[7]={}
    data[7]['name']='mnist-35-39'
    data[7]['ncla']=5
    data[8]={}
    data[8]['name']='mnist-40-44'
    data[8]['ncla']=5
    data[9]={}
    data[9]['name']='mnist-45-49'
    data[9]['ncla']=5
    '''
    alphabets=["Angelic", "Atemayar_Qelisayer", "Atlantean", "Aurek-Besh", "Avesta","Ge_ez,Glagolitic","Gurmukhi","Kannada","Keble","Malayalam","Manipuri","Mongolian","Old_Church_Slavonic_(Cyrillic)","Oriya",
        "Sylheti","Syriac_(Serto)","Tengwar","Tibetan","ULOG","Alphabet_of_the_Magi","Anglo-Saxon_Futhorc","Arcadian","Armenian","Asomtavruli_(Georgian)","Balinese",
        "Bengali","Blackfoot_(Canadian_Aboriginal_Syllabics)","Braille","Burmese_(Myanmar)","Cyrillic","Early_Aramaic","Futurama","Grantha","Greek","Gujarati","Hebrew",
        "Inuktitut_(Canadian_Aboriginal_Syllabics)","Japanese_(hiragana)","Japanese_(katakana)","Korean","Latin","Malay_(Jawi_-_Arabic)","Mkhedruli_(Georgian)","N_Ko",
        "Ojibwe_(Canadian_Aboriginal_Syllabics)","Sanskrit","Syriac_(Estrangelo)","Tagalog","Tifinagh"]
    '''  
    for i in range(10):
            data[i]['background']={'x': [],'y': []}
    for s in ['background','eval']:
        # load one set
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        
        for image,target in loader:
            counter_samples =counter_samples + 1
            label=target.numpy()
            if label<40:
                data[0]['background']['x'].append(image)
                data[0]['background']['y'].append(label)
            if label>=40 and label<80:
                data[1]['background']['x'].append(image)
                data[1]['background']['y'].append(label-5)

            if label>=80 and label<120:
                data[2]['background']['x'].append(image)
                data[2]['background']['y'].append(label-10)
                
            if label>=120 and label<160:
                data[3]['background']['x'].append(image)
                data[3]['background']['y'].append(label-15)
                
            if label>=160 and label<240:
                data[4]['background']['x'].append(image)
                data[4]['background']['y'].append(label-20)
                
            if label>=240 and label<280:
                data[5]['background']['x'].append(image)
                data[5]['background']['y'].append(label-25)
                
            if label>=280 and label<360:
                data[6]['background']['x'].append(image)
                data[6]['background']['y'].append(label-30)
                
            if label>=360 and label<400:
                data[7]['background']['x'].append(image)
                data[7]['background']['y'].append(label-35)

            if label>=400 and label<440:
                data[8]['background']['x'].append(image)
                data[8]['background']['y'].append(label-40)
                
            else:
                data[9]['background']['x'].append(image)
                data[9]['background']['y'].append(label-45)

    # "Unify" and save
    for n in range(10):
        data[n]['background']['x']=torch.stack(data[n]['background']['x']).view(-1,size[0],size[1],size[2])
        data[n]['background']['y']=torch.LongTensor(np.array(data[n]['background']['y'],dtype=int)).view(-1)
        
    # Split background - validation
    l = data[0]['background']['x'].size()[0]
    for t in data.keys():
        data[t]['valid']={}
        data[t]['train']={}
        data[t]['train']['x'],data[t]['valid']['x']=torch.split(data[t]['background']['x'],[int(0.7*l),int(0.3*l)])
        data[t]['train']['y'],data[t]['valid']['y']=torch.split(data[t]['background']['y'],[int(0.7*l),int(0.3*l)])


    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, 5))
        n+=5
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################
