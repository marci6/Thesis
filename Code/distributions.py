# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:59:19 2021

@author: MARCELLOCHIESA
"""
import math
import torch

class VariationalPosterior(torch.nn.Module):
    def __init__(self, mu, rho, device):
        super(VariationalPosterior, self).__init__()
        self.mu = mu.to(device)
        self.rho = rho.to(device)
        self.device = device
        # gaussian distribution to sample epsilon from
        self.normal = torch.distributions.Normal(0, 1)
        # log1p is more accurate than log for small x
        self.sigma = torch.log1p(torch.exp(self.rho)).to(self.device)

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.device)
        # reparametrizarion trick for sampling from posterior
        # epsilon = sample
        posterior_sample = (self.mu + self.sigma * epsilon).to(self.device)
        return posterior_sample

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class Prior(torch.nn.Module):
    '''
    Scaled Gaussian Mixtures for Priors
    '''
    def __init__(self, std1, std2, pi, device):
        super(Prior, self).__init__()
        self.std1 = std1
        self.std2 = std2
        self.pi = pi
        self.device = device

        self.s1 = torch.tensor([math.exp(-1. * self.std1)], dtype=torch.float32, device=self.device)
        self.s2 = torch.tensor([math.exp(-1. * self.std2)], dtype=torch.float32, device=self.device)

        self.gaussian1 = torch.distributions.Normal(0,self.s1)
        self.gaussian2 = torch.distributions.Normal(0,self.s2)


    def log_prob(self, input):
        input = input.to(self.device)
        # probabilities
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1.-self.pi) * prob2)).sum()