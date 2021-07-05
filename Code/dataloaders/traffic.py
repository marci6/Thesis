import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
import pickle
from PIL import Image

########################################################################################################################

# MNIST divided into 10 sets.

def get(data_path,seed,fixed_order=False,pc_valid=0):
    data={}
    taskcla=[]
    size=[3,32,32]

    # Traffic signs
    mean=[0.3398,0.3117,0.3210]
    std=[0.2755,0.2647,0.2712]
    dat={}
    dat['train']=TrafficSigns(os.path.join(data_path, '../data/traffic_signs'), train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=TrafficSigns(os.path.join(data_path, '../data/traffic_signs'), train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data[0]={}
    data[0]['name']='traffic-1-4'
    data[0]['ncla']=4
    
    data[1]={}
    data[1]['name']='traffic-5-8'
    data[1]['ncla']=4
    
    data[2]={}
    data[2]['name']='traffic-9-12'
    data[2]['ncla']=4
    
    data[3]={}
    data[3]['name']='traffic-13-16'
    data[3]['ncla']=4
    
    data[4]={}
    data[4]['name']='traffic-17-20'
    data[4]['ncla']=4
    
    data[5]={}
    data[5]['name']='traffic-21-24'
    data[5]['ncla']=4
    
    data[6]={}
    data[6]['name']='traffic-25-28'
    data[6]['ncla']=4
    
    data[7]={}
    data[7]['name']='traffic-29-32'
    data[7]['ncla']=4
    
    data[8]={}
    data[8]['name']='traffic-33-36'
    data[8]['ncla']=4
    
    data[9]={}
    data[9]['name']='traffic-37-40'
    data[9]['ncla']=4
    
    data[10]={}
    data[10]['name']='traffic-41-43'
    data[10]['ncla']=3
    
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1, shuffle=False)
        data[0][s]={'x': [],'y': []}
        data[1][s]={'x': [],'y': []}
        data[2][s]={'x': [],'y': []}
        data[3][s]={'x': [],'y': []}
        data[4][s]={'x': [],'y': []}
        data[5][s]={'x': [],'y': []}
        data[6][s]={'x': [],'y': []}
        data[7][s]={'x': [],'y': []}
        data[8][s]={'x': [],'y': []}
        data[9][s]={'x': [],'y': []}
        data[10][s]={'x':[],'y': []}
        
        for image,target in loader:
            label=target.numpy()[0]
            
            data[label//4][s]['x'].append(image)
            data[label//4][s]['y'].append(label%4)
        
            # if label<4:
            #     data[0][s]['x'].append(image)
            #     data[0][s]['y'].append(label)
                
            # if label>3 and label <8:
            #     data[1][s]['x'].append(image)
            #     data[1][s]['y'].append(label-4)

            # if label>7 and label <12:
            #     data[2][s]['x'].append(image)
            #     data[2][s]['y'].append(0)
                
            # if label>5 and label <9:
            #     data[3][s]['x'].append(image)
            #     data[3][s]['y'].append(1)

            # if label>5 and label <9:
            #     data[4][s]['x'].append(image)
            #     data[4][s]['y'].append(0)
            
            # if label>5 and label <9:
            #     data[5][s]['x'].append(image)
            #     data[5][s]['y'].append(1)

            # if label>5 and label <9:
            #     data[6][s]['x'].append(image)
            #     data[6][s]['y'].append(0)
            
            # if label>5 and label <9:
            #     data[7][s]['x'].append(image)
            #     data[7][s]['y'].append(1)

            
            # if label>5 and label <9:
            #     data[8][s]['x'].append(image)
            #     data[8][s]['y'].append(0)
                
            # if label>5 and label <9:
            #     data[9][s]['x'].append(image)
            #     data[9][s]['y'].append(1)
                
            # if label>5 and label <9:
            #     data[10][s]['x'].append(image)
            #     data[10][s]['y'].append(1)
      
    # "Unify" and save
    for n in range(11):
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

class TrafficSigns(torch.utils.data.Dataset):
    """`German Traffic Signs <http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.

    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "traffic_signs_dataset.zip"
        self.url = "https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip"
        # Other options for the same 32x32 pickled dataset
        # url="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip"
        # url_train="https://drive.google.com/open?id=0B5WIzrIVeL0WR1dsTC1FdWEtWFE"
        # url_test="https://drive.google.com/open?id=0B5WIzrIVeL0WLTlPNlR2RG95S3c"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                self.download()

        training_file = 'lab 2 data/train.p'
        testing_file = 'lab 2 data/test.p'
        if train:
            with open(os.path.join(root,training_file), mode='rb') as f:
                train = pickle.load(f)
            self.data = train['features']
            self.labels = train['labels']
        else:
            with open(os.path.join(root,testing_file), mode='rb') as f:
                test = pickle.load(f)
            self.data = test['features']
            self.labels = test['labels']

        self.data = np.transpose(self.data, (0, 3, 1, 2))
        #print(self.data.shape); sys.exit()

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)
        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)
        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

