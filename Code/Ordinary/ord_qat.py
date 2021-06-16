# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:59:21 2021

@author: MARCELLOCHIESA
"""

import time
import os
import numpy as np
import copy
import math
import torch
import torch.nn.functional as F
from utils import tb_setup

class Appr(object):

    def __init__(self,model,args,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=1000):
        self.model=model
        self.device = args.device
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.init_lr=args.lr
        self.sbatch=args.sbatch
        self.nepochs= args.nepochs

        self.arch=args.arch
        self.samples=args.samples
        self.lambda_=1.

        self.output=args.output
        self.checkpoint = args.checkpoint
        self.experiment=args.experiment
        self.num_tasks=args.num_tasks

        self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
        self.modules_names_without_cls = self.find_modules_names(with_classifier=False)



    def train(self,t,xtrain,ytrain,xvalid,yvalid):

        self.model.train()
        # attach a global qconfig, which contains information about what kind
        # of observers to attach.
        if t==0:
            self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            self.model_qat = torch.quantization.prepare_qat(self.model, inplace=False)
        # Set optimizer
        params_dict = []
        params_dict.append({'params': self.model_qat.parameters(), 'lr': self.init_lr})
        self.optimizer = torch.optim.Adam(params_dict, lr=self.init_lr)
        best_loss=np.inf

        # best_model=copy.deepcopy(self.model)
        best_model = copy.deepcopy(self.model.state_dict())
        lr = self.init_lr
        patience = self.lr_patience
        
        # Set up TensorBoard
        tb = tb_setup(xtrain, ytrain, self.model, 'ordinary',t)
        
        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train step ###############################
                clock0=time.time()
                
                self.train_epoch(t,xtrain,ytrain)
                
                clock1=time.time()
                train_loss,train_acc = self.eval(t,xtrain,ytrain)
                clock2=time.time()
                
                # Update TensorBoard
                tb.add_scalar('Loss', train_loss, e)
                tb.add_scalar('Accuracy', train_acc, e)
                
                for name, value in self.model.named_parameters():
                    tb.add_histogram(name, value, e )
                    if value.grad is None:
                        pass
                    else:
                        tb.add_histogram('{}.grad'.format(name), value.grad, e)
                
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),
                    train_loss,100*train_acc),end='')
                # Valid accuracy
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

                if math.isnan(valid_loss) or math.isnan(train_loss):
                    print("saved best model and quit because loss became nan")
                    break

                # Adapt lr
                if valid_loss<best_loss:
                    # save best loss and best model
                    best_loss=valid_loss
                    best_model=copy.deepcopy(self.model_qat.state_dict())
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        # decrease lr
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        # update optimizer
                        self.optimizer=torch.optim.Adam(params_dict, lr=lr)

                print()
        except KeyboardInterrupt:
            print()
        
        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, fuses modules where appropriate,
        # and replaces key operators with quantized implementations.
        if t==self.num_tasks-1:
            self.model = torch.quantization.convert(self.model_qat.eval(), inplace=False)
        # Close TensorBoard
        tb.close()
        # Restore best
        # self.model.load_state_dict(copy.deepcopy(torch.quantization.convert(best_model.eval(), inplace=False)))
        self.save_model(t)


    def find_modules_names(self, with_classifier=False):
        modules_names = []
        for name, p in self.model.named_parameters():
            if with_classifier is False:
                if not name.startswith('classifier'):
                    n = name.split('.')[:-1]
                    modules_names.append('.'.join(n))
            else:
                n = name.split('.')[:-1]
                modules_names.append('.'.join(n))

        modules_names = set(modules_names)

        return modules_names


    def train_epoch(self,t,x,y):

        self.model_qat.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(self.device)
#        j=0
        # Loop batches
        for i in range(0,len(r),self.sbatch):
            # b -> batch
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images, targets = x[b].to(self.device), y[b].to(self.device)
            # Forward
            predictions = self.model_qat(images)[t]
            # Compute loss
            loss = torch.nn.functional.nll_loss(predictions, targets).to(self.device)

            # Backward
            # self.model.cuda()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # self.model.cuda()

            # Update parameters
            self.optimizer.step()
        return


    def eval(self,t,x,y,debug=False):
        total_loss=0
        total_acc=0
        total_num=0
        self.model_qat.eval()

        r=np.arange(x.size(0))
        r=torch.as_tensor(r, device=self.device, dtype=torch.int64)

        with torch.no_grad():
            # Loop batches
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images, targets = x[b].to(self.device), y[b].to(self.device)

                # Forward
                predictions = self.model_qat(images)[t]
                # Compute loss
                loss = torch.nn.functional.nll_loss(predictions, targets).to(device=self.device)
                # take output with highest probability
                _,pred=predictions.max(1, keepdim=True)

                total_loss += loss.detach()*len(b)
                total_acc += pred.eq(targets.view_as(pred)).sum().item() 
                total_num += len(b)           

        return total_loss/total_num, total_acc/total_num


    def set_model_(model, state_dict):
        model.model.load_state_dict(copy.deepcopy(state_dict))


    def save_model(self,t):
        torch.save({'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(t)))



