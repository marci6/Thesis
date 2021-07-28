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

if args.qat:
    torch.backends.quantized.engine = 'fbgemm'

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
elif args.experiment == 'omniglot':
    from dataloaders import omniglot as dataloader
elif args.experiment == 'fmnist':
    from dataloaders import fmnist as dataloader
elif args.experiment == 'easymix':
    from dataloaders import easymix as dataloader
elif args.experiment == 'SVHN':
    from dataloaders import SVHN as dataloader
elif args.experiment == 'cSVHN':
    from dataloaders import SVHN_complete as dataloader
elif args.experiment == 'mnistm':
    from dataloaders import mnistm as dataloader
elif args.experiment == 'cifar10':
    from dataloaders import cifar10 as dataloader
elif args.experiment == 'cifar5':
    from dataloaders import cifar5 as dataloader
elif args.experiment == 'traffic':
    from dataloaders import traffic as dataloader
elif args.experiment == 'cfmnist':
    from dataloaders import fmnist_complete as dataloader

# Args -- Approach
if args.approach =='ucb':
    if args.arch == 'mlp':
        if args.qat:
            from Networks import UCB_qat as approach
        else:
            from Networks import UCB as approach
        # Args -- Network
        
        if args.qat:
            from Networks import MLP_quantized as network
        else:
            from Networks import MLP as network
    else:
        if args.qat:
            from Ordinary import ord_qat as approach
        else:
            from Ordinary import ordinary as approach
        # Args -- Network
        if args.qat:
            from Ordinary import resnet_qat as network   
        else:
            from Ordinary import resnet as network 
        
elif args.approach =='ord':
    if args.arch == 'mlp':
        if args.qat:
            from Ordinary import ord_qat as approach
        else:
            from Ordinary import ordinary as approach
        # Args -- Network
        from Ordinary import MLP as network
    else:
        if args.qat:
            from Networks import UCB_qat as approach
        else:
            from Networks import UCB as approach
        # Args -- Network
        if args.qat:
            from Networks import resnet_ucb_qat as network   
        else:
            from Networks import resnet_ucb as network 
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

print('-'*20)
appr = approach.Appr(model,args=args)
print('-'*20)

args.output=os.path.join(args.results_path, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
print('-'*20)

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
    
    print('*'*20)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*20)

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
        xtrain=data[t]['train']['x']
        ytrain=data[t]['train']['y']
        xvalid=data[t]['valid']['x']
        yvalid=data[t]['valid']['y']
        task=t

    # Train
    appr.train(task,xtrain,ytrain,xvalid,yvalid)
    model = appr.model
    print('-'*20)

    # Test model on all previous tasks, after learning current task
    for u in range(t+1):
        xtest=data[u]['test']['x'].to(args.device)
        ytest=data[u]['test']['y'].to(args.device)
        test_loss,test_acc = appr.eval(u,xtest,ytest,debug=True)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.3f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

    # Save
    print('Save at ' + args.checkpoint)
    np.savetxt(os.path.join(args.checkpoint,'{}_{}_{}.txt'.format(args.experiment,args.approach,args.seed)),acc,'%.5f')


utils.print_log_acc_bwt(args, acc, lss)
print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))
#%%
print('\nPre quantization')
pre_size = utils.print_size_of_model(model)

if args.qat:
    torch.quantization.convert(model.eval().to('cpu'), inplace=True)
    print('Post quantization')
    post_size = utils.print_size_of_model(model)
    print('\nQuantized model is X {:.3f} smaller'.format(pre_size/post_size))

if args.save_model:
    PATH = os.path.join(args.save_path,'{}_{}'.format(args.experiment,args.approach))
    if not os.path.isdir(args.save_path):
        os.makedirs(PATH)
    if not os.path.exists(checkpoint):
        os.mkdir(PATH)
    torch.save(model.state_dict(), PATH)

############ QUANTIZE ###############################################################################
if args.dynamic:
    
    qmodel = utils.quantize(model.to('cpu'), torch.qint8)
    
    print()
    post_size = utils.print_size_of_model(qmodel)
    
    print('\nQuantized model is X {:.1f} smaller'.format(pre_size/post_size))

################ CHECK ACCURACY #############################################
    qacc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    qlss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    appr.model = qmodel
    if args.device == 'cuda:0':
        appr.device = 'cpu'
        # Test model on all previous tasks, after learning current task
        for t,ncla in taskcla[args.sti:]:
            for u in range(t+1):
                xtest=data[u]['test']['x'].to(args.device)
                ytest=data[u]['test']['y'].to(args.device)
                test_loss,test_acc=appr.eval(u,xtest,ytest,debug=True)
                qacc[t,u]=test_acc
                qlss[t,u]=test_loss
                
        utils.print_log_acc_bwt(args, qacc, qlss)
        appr.device = 'cuda:0'
    else:
        # Test model on all previous tasks, after learning current task
        for t,ncla in taskcla[args.sti:]:
            for u in range(t+1):
                xtest=data[u]['test']['x'].to(args.device)
                ytest=data[u]['test']['y'].to(args.device)
                test_loss,test_acc=appr.eval(u,xtest,ytest,debug=True)
                qacc[t,u]=test_acc
                qlss[t,u]=test_loss
                
        utils.print_log_acc_bwt(args, qacc, qlss)
#%%
utils.acc_graph(args,acc)

######## CHECK INFERENCE TIME ###########
mean_syn, std_syn = utils.inference_time(model, args.inputsize)
print('\nThe mean inference time is {:5.3f}ms and the std is {:5.3f}ms'.format(mean_syn, std_syn))

