#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   bpsvi_gaussian_hvae.py
    @Time    :   2021/11/02 19:42:36
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   None
'''
import torch
import math

log_2_pi = math.log(2*math.pi)


# ---- Define Coreset Settings ----#
def sample_z(prob_w, pts, cur_model, cur_device):
    index_pts = torch.multinomial(prob_w, 500, replacement=True)
    q_z2_mean, q_z2_logvar = cur_model.q_z(pts[index_pts], prior=True)
    z2 = cur_model.reparameterize(q_z2_mean, q_z2_logvar)
    return z2


def log_likelihood(pts, samples, cur_model):
    z1_sample_mean, z1_sample_logvar = cur_model.p_z1(samples)
    z1_sample_rand = cur_model.reparameterize(z1_sample_mean, z1_sample_logvar)
    x_mean, _ = cur_model.p_x(z1_sample_rand.view(-1, z1_sample_mean.shape[1]),
                              samples.view(-1, samples.shape[1]))

    # x_mean, x_logvar = cur_model.p_x(samples)
    log_normal = -0.5 * (log_2_pi + torch.pow(pts.unsqueeze(1) - x_mean, 2))
    log_normal = torch.sum(log_normal, dim=2)
    return log_normal


def grad_log_likelihood(pts, samples, cur_model, cur_device):
    z1_sample_mean, z1_sample_logvar = cur_model.p_z1(samples)
    z1_sample_rand = cur_model.reparameterize(z1_sample_mean, z1_sample_logvar)
    x_mean, _ = cur_model.p_x(z1_sample_rand.view(-1, z1_sample_mean.shape[1]),
                              samples.view(-1, samples.shape[1]))
    # x_mean, x_logvar = cur_model.p_x(samples)
    grad_log_likelihood = - (pts.unsqueeze(1) - x_mean)
    return grad_log_likelihood


class BlackBoxProjector_HVAE(object):
    def __init__(self, sampler=sample_z, projection_dimension=200, loglikelihood=log_likelihood,
                 grad_loglikelihood=grad_log_likelihood, device='cpu'):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.device = device

    def project(self, pts, model, grad=False):
        lls = self.loglikelihood(pts, self.samples, model)  # B*S  M*S
        lls = lls - lls.mean(dim=1).unsqueeze(1)
        if grad:
            if self.grad_loglikelihood is None:
                raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            glls = self.grad_loglikelihood(pts, self.samples, model, cur_device=self.device)  # M*S*d
            glls = glls - glls.mean(dim=2).unsqueeze(2)
            return lls, glls
        else:
            return lls

    def update(self, prob_w, pts, model):
        self.samples = self.sampler(prob_w, pts, model, self.device)