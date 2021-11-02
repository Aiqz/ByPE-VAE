#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   Coreset.py
    @Time    :   2021/11/02 15:30:04
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   Coreset class
'''
import torch
import numpy as np


class Coreset(object):
    def __init__(self, train_loader, coreset_size=500, init_use_data='False', device='cpu', step_sched=0.1):
        self.train_loader = train_loader
        self.coreset_size = coreset_size
        self.device = device
        self.pts, self.wts = self.generate_coreset(init_use_data)
        self.x0 = torch.cat((self.wts, self.pts.reshape(self.wts.shape[0] * self.pts.shape[1])), dim=0)
        self.nn_idcs = torch.arange(self.wts.shape[0])
        self.x = self.x0.clone()

        self.m1 = torch.zeros(self.x0.shape[0]).to(self.device)
        self.m2 = torch.zeros(self.x0.shape[0]).to(self.device)

        self.step_sched = lambda i: step_sched/(i+1)

        self.best_wts = self.wts
        self.best_pts = self.pts

    def generate_coreset(self, init_use_data):
        if init_use_data:
            dataset = self.train_loader.dataset
            p = dataset[torch.randint(0, len(self.train_loader.dataset), (self.coreset_size,))][0].to(self.device)
            w = len(self.train_loader.dataset) / self.coreset_size * torch.ones(self.coreset_size).to(self.device)
        else:
            p = torch.normal(mean=0., std=0.01, size=(self.coreset_size, 784)).to(self.device)
            w = len(self.train_loader.dataset) / self.coreset_size * torch.ones(self.coreset_size).to(self.device)
        return p, w

    def update(self, new_p, new_w):
        self.pts = new_p
        self.wts = new_w