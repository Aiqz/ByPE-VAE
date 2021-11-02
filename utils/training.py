from __future__ import print_function
import torch


def set_beta(args, epoch):
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1. * epoch / args.warmup
        if beta > 1.:
            beta = 1.
    return beta


def train_one_epoch(epoch, args, train_loader, model, optimizer):
    train_loss, train_re, train_kl = 0, 0, 0
    model.train()
    beta = set_beta(args, epoch)
    print('beta: {}'.format(beta))
    if args.approximate_prior is True:
        with torch.no_grad():
            cached_z, cached_log_var = model.cache_z(train_loader.dataset)
            cache = (cached_z, cached_log_var)
    else:
        cache = None

    for batch_idx, (data, indices, target) in enumerate(train_loader):
        data, indices, target = data.to(args.device), indices.to(args.device), target.to(args.device)

        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        x = (x, indices)
        optimizer.zero_grad()

        loss, RE, KL = model.calculate_loss(x, beta, average=True, cache=cache, dataset=train_loader.dataset)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.data.item()
            train_re +=   -RE.data.item()
            train_kl +=   KL.data.item()
            if cache is not None:
                cache = (cache[0].detach(), cache[1].detach())
    train_loss /= len(train_loader)
    train_re /= len(train_loader)
    train_kl /= len(train_loader)
    return train_loss, train_re, train_kl


def train_one_epoch_with_CE_proir(epoch, args, train_loader, model, optimizer, coreset=None):
    train_loss, train_re, train_kl = 0, 0, 0
    model.train()
    beta = set_beta(args, epoch)
    print('beta: {}'.format(beta))

    for batch_idx, data in enumerate(train_loader):
        if len(data) == 3:
            data, indices, target = data
            data, indices, target = data.to(args.device), indices.to(args.device), target.to(args.device)
        else:
            data, target = data
            data, target = data.to(args.device), target.to(args.device)
            indices = None

        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        if args.dataset_name == 'celeba':
            x = x.view(-1, 64 * 64 * 3)

        x = (x, indices)
        optimizer.zero_grad()

        loss, RE, KL = model.calculate_loss(x, beta, average=True, cache=None, dataset=coreset)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.data.item()
            train_re +=   -RE.data.item()
            train_kl +=   KL.data.item()
    train_loss /= len(train_loader)
    train_re /= len(train_loader)
    train_kl /= len(train_loader)
    return train_loss, train_re, train_kl


def update_coreset(coreset, args, times, model, train_loader, prj_z):
    model.eval()
    # parameters in nn_opt
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    with torch.no_grad():
        for batch_idx, (data, indices, target) in enumerate(train_loader):
            data, indices, target = data.to(args.device), indices.to(args.device), target.to(args.device)

            # if args.dynamic_binarization:
            #     x = torch.bernoulli(data)
            # else:
            #     x = data
            x = data
            x = (x, indices)
            inputs, _ = x

            w = coreset.wts
            p = coreset.pts

            # prob_w = w / len(coreset.train_loader.dataset)
            average_w = args.training_set_size / args.coreset_size
            w_bias = average_w - w.mean()
            prob_w = (w + w_bias) / len(coreset.train_loader.dataset)

            sz = w.shape[0]
            d = p.shape[1]

            vecs, sum_scaling, corevecs, pgrads = model.get_projection(model, prj_z, inputs, p, prob_w)

            vecs = vecs.detach()
            sum_scaling = len(train_loader.dataset) / inputs.shape[0]
            corevecs = corevecs.detach()
            pgrads = pgrads.detach()

            # compute gradient of weights and pts
            resid = sum_scaling * vecs.sum(axis=0) - torch.matmul(w, corevecs)
            wgrad = -torch.matmul(corevecs, resid) / corevecs.shape[1]
            ugrad = -(w.unsqueeze(-1).unsqueeze(-1) * pgrads * resid.unsqueeze(0).unsqueeze(-1)).sum(dim=1) / corevecs.shape[1]

            g = torch.cat((wgrad, ugrad.reshape(sz * d)), dim=0)

            j = batch_idx + times * ((len(train_loader.dataset)) / inputs.shape[0])
            coreset.m1 = b1 * coreset.m1 + (1. - b1) * g
            coreset.m2 = b2 * coreset.m2 + (1. - b2) * g ** 2
            upd = coreset.step_sched(j) * coreset.m1 / (1. - b1 ** (j + 1)) / (
                        eps + torch.sqrt(coreset.m2 / (1. - b2 ** (j + 1))))
            coreset.x -= upd
            # project onto x>=0
            if coreset.nn_idcs is None:
                coreset.x = torch.max(coreset.x, torch.zeros_like(coreset.x).to(args.device))

            else:
                coreset.x[coreset.nn_idcs] = torch.max(coreset.x[coreset.nn_idcs],
                                                       torch.zeros_like(coreset.x[coreset.nn_idcs]).to(args.device))
                # coreset.x[sz:] = torch.clamp(coreset.x[sz:], 0., 1.)

            coreset.wts = coreset.x[:sz]
            coreset.pts = coreset.x[sz:].reshape((sz, d))
    print("Coreset update!")
