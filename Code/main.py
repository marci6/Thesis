# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:08:36 2021

@author: MARCELLOCHIESA
"""
import os
import time
import numpy as np
import torch
import utils
from datetime import datetime
from param import Parameters


tstart=time.time()

# Arguments

args = Parameters()

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print('Using device:', args.device)
checkpoint = utils.make_directories(args)
args.checkpoint = checkpoint
print()


# Args -- Experiment
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='mnist5':
    from dataloaders import mnist5 as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach =='ucb':
    import UCB as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment == 'mnist5':
    import MLP as network
else:
    import resnet_ucb as network

########################################################################################################################
print()
print("Starting this run on :")
print(datetime.now().strftime("%Y-%m-%d %H:%M"))

# Load
print('Load data...')
data,taskcla,inputsize = dataloader.get(data_path=args.data_path, seed=args.seed)
print('Input size =',inputsize,'\nTask info =',taskcla)
args.num_tasks=len(taskcla)
args.inputsize, args.taskcla = inputsize, taskcla

# Inits
print('Inits...')
model=network.Net(args).to(args.device)
print()
print('Printing current model...')
print(model)
print()

print('-'*50)
appr=approach.Appr(model,args=args)
print('-'*50)

args.output=os.path.join(args.results_path, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
print('-'*50)

if args.resume == 'yes':
    checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(args.sti)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=args.device)
else:
    args.sti = 0

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla[args.sti:]:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t==0:
            # data x-y
            xtrain=data[t]['train']['x']
            ytrain=data[t]['train']['y']
            xvalid=data[t]['valid']['x']
            yvalid=data[t]['valid']['y']
            # task zero array
            task_t=t*torch.ones(xtrain.size(0)).int()
            task_v=t*torch.ones(xvalid.size(0)).int()
            task=[task_t,task_v]
        else:
            # data x-y
            xtrain=torch.cat((xtrain,data[t]['train']['x']))
            ytrain=torch.cat((ytrain,data[t]['train']['y']))
            xvalid=torch.cat((xvalid,data[t]['valid']['x']))
            yvalid=torch.cat((yvalid,data[t]['valid']['y']))
            # task array
            task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
            task=[task_t,task_v]
    else:
        # Get data
        xtrain=data[t]['train']['x'].to(args.device)
        ytrain=data[t]['train']['y'].to(args.device)
        xvalid=data[t]['valid']['x'].to(args.device)
        yvalid=data[t]['valid']['y'].to(args.device)
        task=t

    # Train
    appr.train(task,xtrain,ytrain,xvalid,yvalid)
    print('-'*50)

    # Test model on all previous tasks, after learning current task
    for u in range(t+1):
        xtest=data[u]['test']['x'].to(args.device)
        ytest=data[u]['test']['y'].to(args.device)
        test_loss,test_acc=appr.eval(u,xtest,ytest,debug=True)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

    # Save
    print('Save at '+args.checkpoint)
    np.savetxt(os.path.join(args.checkpoint,'{}_{}_{}.txt'.format(args.experiment,args.approach,args.seed)),acc,'%.5f')


utils.print_log_acc_bwt(args, acc, lss)
print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

