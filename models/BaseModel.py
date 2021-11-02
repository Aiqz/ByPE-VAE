from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from utils.nn import normal_init, NonLinear
from utils.distributions import log_normal_diag_vectorized, log_normal_diag_vectorized_ce
import math
from utils.nn import he_init
from utils.distributions import pairwise_distance
from utils.distributions import log_bernoulli, log_normal_diag, log_normal_standard, log_logistic_256
from abc import ABC, abstractmethod
from models.Coreset import Coreset
import torch.nn.functional as F


class BaseModel(nn.Module, ABC):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        print("constructor")
        self.args = args
        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

        if self.args.prior == 'exemplar_prior':
            self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))

        if self.args.prior == 'CE_prior':
            self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())

        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size))
            self.p_x_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size),
                                        activation=nn.Hardtanh(min_val=-4.5, max_val=0))
            self.decoder_logstd = torch.nn.Parameter(torch.tensor([0.], requires_grad=True))
        self.create_model(args)
        self.he_initializer()

    def he_initializer(self):
        print("he initializer")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    @abstractmethod
    def create_model(self, args):
        pass

    @abstractmethod
    def kl_loss(self, latent_stats, exemplars_embedding, dataset, cache, x_indices, training_size):
        pass

    def reconstruction_loss(self, x, x_mean, x_logvar):
        if self.args.input_type == 'binary':
            return log_bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            if self.args.use_logit is True:
                return log_normal_diag(x, x_mean, x_logvar, dim=1)
            else:
                return log_logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

    def calculate_loss(self, x, beta=1., average=False,
                       exemplars_embedding=None, cache=None, dataset=None):
        x, x_indices = x
        x_mean, x_logvar, latent_stats = self.forward(x)
        # RE = self.reconstruction_loss(x, x_mean, x_logvar)
        RE = -F.mse_loss(x, x_mean, reduction='none')
        RE = torch.sum(RE, dim=1)
        KL = self.kl_loss(latent_stats, exemplars_embedding, dataset, cache, x_indices, self.args.training_set_size)
        # loss = -RE + beta * KL
        # loss = RE + beta * KL
        if average:
            # loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        loss = -RE + beta * KL
        return loss, RE, KL

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = mu.new_empty(size=std.shape).normal_()
        return eps.mul(std).add_(mu)

    def log_p_z_vampprior(self, z, exemplars_embedding):
        if exemplars_embedding is None:
            C = self.args.number_components
            X = self.means(self.idle_input)
            z_p_mean, z_p_logvar = self.q_z(X, prior=True)  # C x M
        else:
            C = torch.tensor(self.args.number_components).float()
            z_p_mean, z_p_logvar = exemplars_embedding

        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)
        return log_normal_diag(z_expand, means, logvars, dim=2) - math.log(C)

    def log_p_z_exemplar(self, z, z_indices, exemplars_embedding, test):
        centers, center_log_variance, center_indices = exemplars_embedding
        denominator = torch.tensor(len(centers)).expand(len(z)).float().to(self.args.device)
        center_log_variance = center_log_variance[0, :].unsqueeze(0)
        prob, _ = log_normal_diag_vectorized(z, centers, center_log_variance)  # MB x C
        if test is False and self.args.no_mask is False:
            mask = z_indices.expand(-1, len(center_indices)) \
                    == center_indices.squeeze().unsqueeze(0).expand(len(z_indices), -1)
            prob.masked_fill_(mask, value=float('-inf'))
            denominator = denominator - mask.sum(dim=1).float()
        prob -= torch.log(denominator).unsqueeze(1)
        return prob

    def log_p_z_ce(self, z, exemplars_embedding, training_size):
        centers, center_log_variance = exemplars_embedding
        # print(training_size)
        denominator = torch.tensor(training_size).expand(len(z)).float().to(self.args.device)
        center_log_variance = center_log_variance[0, :].unsqueeze(0)
        # prob, _ = log_normal_diag_vectorized(z, centers, center_log_variance, wts)  # MB x C
        first_part, second_part = log_normal_diag_vectorized_ce(z, centers, center_log_variance)

        third_part = -torch.log(denominator).unsqueeze(1)
        return first_part, second_part, third_part

    def log_p_z_core_exemplar(self, z, exemplars_embedding):

        # C= torch.tensor(self.args.number_components).float()
        z_p_mean, z_p_logvar = exemplars_embedding
        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)
        ss = self.stat_n
        stat_n = self.stat_n.unsqueeze(0)
        n=log_normal_diag(z_expand, means, logvars, dim=2)

        return log_normal_diag(z_expand, means, logvars, dim=2) + torch.log(stat_n)

    def log_p_z(self, z, exemplars_embedding, sum=True, test=None, training_size=50000, wts=None):
        z, z_indices = z
        if test is None:
            test = not self.training
        if self.args.prior == 'standard':
            return log_normal_standard(z, dim=1)
        elif self.args.prior == 'vampprior':
            prob = self.log_p_z_vampprior(z, exemplars_embedding)
        elif self.args.prior == 'exemplar_prior':
            prob = self.log_p_z_exemplar(z, z_indices, exemplars_embedding, test)
        elif self.args.prior == 'CE_prior':
            first_part, second_part, third_part = self.log_p_z_ce(z, exemplars_embedding, training_size=training_size)
            prob = first_part + second_part + third_part
            # prob = torch.mul(prob, wts)
            # print(torch.max(prob))
        elif self.args.prior == 'coreset_exemplar_prior':
            prob = self.log_p_z_core_exemplar(z, exemplars_embedding)

        else:
            raise Exception('Wrong name of the prior!')
        if sum:
            if self.args.prior == 'CE_prior':
                second_part += torch.log(wts)

                prob = first_part + second_part + third_part
                prob_max, _ = torch.max(prob, 1)
                # exp_max = torch.exp(-prob_max.unsqueeze(1))
                # exp_all = exp_f * exp_s * exp_t * exp_max
                exp_all = torch.exp(prob - prob_max.unsqueeze(1))

                log_prior = prob_max + torch.log(torch.sum(exp_all, 1))
            else:
                prob_max, _ = torch.max(prob, 1)  # MB x 1
                # print("prob_max:" + str(prob_max.shape))
                log_prior = prob_max + torch.log(torch.sum(torch.exp(prob - prob_max.unsqueeze(1)), 1))  # MB x 1
                # print(torch.max(log_prior))
        else:
            return prob
        return log_prior

    def add_pseudoinputs(self):
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False, activation=nonlinearity)
        # init pseudo-inputs
        if self.args.use_training_data_init:
            print(self.args.pseudoinputs_mean)
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)
        self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components), requires_grad=False)
        self.idle_input = self.idle_input.to(self.args.device)

    def generate_z_interpolate(self, exemplars_embedding=None, dim=0):
        new_zs = []
        exemplars_embedding, _, _ = exemplars_embedding
        step_counts = 10
        step = (exemplars_embedding[1] - exemplars_embedding[0])/step_counts
        for i in range(step_counts):
            new_z = exemplars_embedding[0].clone()
            new_z += i*step
            new_zs.append(new_z.unsqueeze(0))
        return torch.cat(new_zs, dim=0)

    def generate_z(self, N=25, dataset=None):
        if self.args.prior == 'standard':
            z_sample_rand = torch.FloatTensor(N, self.args.z1_size).normal_().to(self.args.device)
        elif self.args.prior == 'vampprior':
            means = self.means(self.idle_input)[0:N]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            z_sample_rand = z_sample_rand.to(self.args.device)

        elif self.args.prior == 'exemplar_prior':
            rand_indices = torch.randint(low=0, high=self.args.training_set_size, size=(N,))
            exemplars = dataset.tensors[0][rand_indices]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(exemplars.to(self.args.device), prior=True)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            z_sample_rand = z_sample_rand.to(self.args.device)

        elif self.args.prior == 'coreset_exemplar_prior':
            rand_indices = torch.distributions.categorical.Categorical(self.stat_n).sample([N])
            exemplars = self.VQ.embed_code(rand_indices)
            logvar = self.prior_log_variance * torch.ones((exemplars.shape[0], self.args.z1_size)).to(self.args.device)
            z_sample_rand = self.reparameterize(exemplars, logvar)

        elif self.args.prior == 'CE_prior':
            exemplars = dataset
            # rand_indices = torch.randint(low=0, high=self.args.coreset_size, size=(1,))

            # exemplars = pts[rand_indices]
            print(exemplars.shape)
            # exemplars = exemplars.repeat(N, 1)
            # print(exemplars.shape)

            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(exemplars.to(self.args.device), prior=True)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            z_sample_rand = z_sample_rand.to(self.args.device)
        return z_sample_rand

    def reference_based_generation_z(self, N=25, reference_image=None):
        pseudo, log_var = self.q_z(reference_image.to(self.args.device), prior=True)
        # pseudo, log_var = self.q_z(reference_image.to(self.args.device))
        pseudo = pseudo.unsqueeze(1).expand(-1, N, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, N, pseudo.shape[1])
        return z_sample_rand

    def reconstruct_x(self, x):
        x_reconstructed, _, z = self.forward(x)
        # if self.args.model_name == 'pixelcnn':
        #     x_reconstructed = self.pixelcnn_generate(z[0].reshape(-1, self.args.z1_size), z[3].reshape(-1, self.args.z2_size))
        return x_reconstructed

    def logit_inverse(self, x):
        sigmoid = torch.nn.Sigmoid()
        lambd = self.args.lambd
        return ((sigmoid(x) - lambd)/(1-2*lambd))

    def generate_x(self, N=25, dataset=None):
        z2_sample_rand = self.generate_z(N=N, dataset=dataset)
        return self.generate_x_from_z(z2_sample_rand)

    def reference_based_generation_x(self, N=25, reference_image=None):
        z2_sample_rand = \
            self.reference_based_generation_z(N=N, reference_image=reference_image)
        generated_x = self.generate_x_from_z(z2_sample_rand)
        return generated_x

    def generate_x_interpolate(self, exemplars_embedding, dim=0):
        zs = self.generate_z_interpolate(exemplars_embedding, dim=dim)
        print(zs.shape)
        return self.generate_x_from_z(zs, with_reparameterize=False)

    def reshape_variance(self, variance, shape):
        return variance[0]*torch.ones(shape).to(self.args.device)

    def q_z(self, x, prior=False):
        if 'conv' in self.args.model_name or 'pixelcnn' == self.args.model_name:
            x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        h = self.q_z_layers(x)
        if 'conv' in self.args.model_name or self.args.model_name == 'pixelcnn':
            h = h.view(x.size(0), -1)
        z_q_mean = self.q_z_mean(h)

        if prior is True:
            if self.args.prior == 'exemplar_prior':
                z_q_logvar = self.prior_log_variance * torch.ones((x.shape[0], self.args.z1_size)).to(self.args.device)
                if self.args.model_name == 'newconvhvae_2level':
                    z_q_logvar = z_q_logvar.reshape(-1, 4, 4, 4)
            elif self.args.prior == 'CE_prior':
                if self.args.model_name == 'single_conv':
                    z_q_logvar = self.prior_log_variance * torch.ones(z_q_mean.shape).to(self.args.device)
                else:
                    z_q_logvar = self.prior_log_variance * torch.ones((x.shape[0], self.args.z1_size)).to(self.args.device)
                if self.args.model_name == 'newconvhvae_2level':
                    z_q_logvar = z_q_logvar.reshape(-1, 4, 4, 4)
            else:
                z_q_logvar = self.q_z_logvar(h)
        else:
            z_q_logvar = self.q_z_logvar(h)
        return z_q_mean.reshape(-1, self.args.z1_size), z_q_logvar.reshape(-1, self.args.z1_size)

    def cache_z(self, dataset, prior=True, cuda=True):
        cached_z = []
        cached_log_var = []
        caching_batch_size = 10000
        num_batchs = math.ceil(len(dataset) / caching_batch_size)
        for i in range(num_batchs):
            if len(dataset[0]) == 3:
                batch_data, batch_indices, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]
            else:
                batch_data, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]

            exemplars_embedding, log_variance_z = self.q_z(batch_data.to(self.args.device), prior=prior)
            cached_z.append(exemplars_embedding)
            cached_log_var.append(log_variance_z)
        cached_z = torch.cat(cached_z, dim=0)
        cached_log_var = torch.cat(cached_log_var, dim=0)
        cached_z = cached_z.to(self.args.device)
        cached_log_var = cached_log_var.to(self.args.device)
        return cached_z, cached_log_var

    def get_coreset_exemplar_set(self, dataset=None):
        if dataset == None:
            exemplars_z = self.VQ.embed.transpose(0, 1)
            log_variance = self.prior_log_variance * torch.ones((self.args.number_coreset, self.args.z1_size)).to(
                self.args.device)
        else:
            exemplars_indices = torch.randint(low=0, high=self.args.training_set_size,
                                                  size=(self.args.number_components,))
            exemplars_z, _ = self.q_z(dataset.tensors[0][exemplars_indices].to(self.args.device), prior=True)
            self.VQ.not_compute_n = False
            embed_onehot = self.VQ(exemplars_z)
            self.VQ.not_compute_n = True
            n = embed_onehot.sum()
            self.stat_n = embed_onehot.sum(0) / n
            exemplars_z = self.VQ.embed.transpose(0, 1)
            log_variance = self.prior_log_variance * torch.ones((exemplars_z.shape[0], exemplars_z.shape[1])).to(self.args.device)

        return (exemplars_z, log_variance)

    def get_exemplar_set(self, z_mean, z_log_var, dataset, cache, x_indices):
        if self.args.approximate_prior is False:
            exemplars_indices = torch.randint(low=0, high=self.args.training_set_size,
                                              size=(self.args.number_components, ))
            exemplars_z, log_variance = self.q_z(dataset.tensors[0][exemplars_indices].to(self.args.device), prior=True)
            exemplar_set = (exemplars_z, log_variance, exemplars_indices.to(self.args.device))
        else:
            exemplar_set = self.get_approximate_nearest_exemplars(
                z=(z_mean, z_log_var, x_indices),
                dataset=dataset,
                cache=cache)
        return exemplar_set

    def get_approximate_nearest_exemplars(self, z, cache, dataset):
        exemplars_indices = torch.randint(low=0, high=self.args.training_set_size,
                                          size=(self.args.number_components, )).to(self.args.device)
        z, _, indices = z
        cached_z, cached_log_variance = cache
        cached_z[indices.reshape(-1)] = z
        sub_cache = cached_z[exemplars_indices, :]
        _, nearest_indices = pairwise_distance(z, sub_cache) \
            .topk(k=self.args.approximate_k, largest=False, dim=1)
        nearest_indices = torch.unique(nearest_indices.view(-1))
        exemplars_indices = exemplars_indices[nearest_indices].view(-1)
        exemplars = dataset.tensors[0][exemplars_indices].to(self.args.device)
        exemplars_z, log_variance = self.q_z(exemplars, prior=True)
        cached_z[exemplars_indices] = exemplars_z
        exemplar_set = (exemplars_z, log_variance, exemplars_indices)
        return exemplar_set

    def get_projection(self, model, ll_projector, batch_x, p, prob_w):
        # update the projector
        ll_projector.update(prob_w, p, model)
        # construct a tangent space
        vecs = ll_projector.project(batch_x, model)
        sum_scaling = 0.
        corevecs, pgrads = ll_projector.project(p, model, grad=True)
        return vecs, sum_scaling, corevecs, pgrads

    def get_ce_set(self, coreset):
        # inputs = torch.clamp(coreset.pts, min=0., max=1.0)
        exemplars_z, log_variance = self.q_z(coreset.pts, prior=True)
        ce_emd = (exemplars_z, log_variance)
        return ce_emd