#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File    :   data_loader.py
    @Time    :   2021/11/02 10:09:59
    @Author  :   Qingzhong Ai 
    @Contact :   aqz1995@163.com
    @Desc    :   Dataloader for mnist, fashion, cifar10, celeba
'''
import os
import glob
import torch
import torch.utils.data as data_utils
import numpy as np
from PIL import Image
from torchvision import datasets
from .base_data_loader import base_load_data


# Dataloader for Dynamic mnist
class dynamic_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(dynamic_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        train = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test = datasets.MNIST(os.path.join('datasets', self.args.dataset_name), train=False)
        return train, test


# Dataloader for Fashion mnist
class fashion_mnist_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(fashion_mnist_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        train = datasets.FashionMNIST(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test = datasets.FashionMNIST(os.path.join('datasets', self.args.dataset_name), train=False)
        return train, test


# Dataloader for cifar10
class cifar10_loader(base_load_data):
    def __init__(self, args, use_fixed_validation=False, no_binarization=False):
        super(cifar10_loader, self).__init__(args, use_fixed_validation, no_binarization=no_binarization)

    def obtain_data(self):
        training_dataset = datasets.CIFAR10(os.path.join('datasets', self.args.dataset_name), train=True, download=True)
        test_dataset = datasets.CIFAR10(os.path.join('datasets', self.args.dataset_name), train=False)
        return training_dataset, test_dataset

    def seperate_data_from_label(self, train_dataset, test_dataset):
        train_data = np.swapaxes(np.swapaxes(train_dataset.data, 1, 2), 1, 3)
        y_train = np.asarray(train_dataset.targets).astype(int)
        test_data = np.swapaxes(np.swapaxes(test_dataset.data, 1, 2), 1, 3)
        y_test = np.asarray(test_dataset.targets).astype(int)
        return train_data, y_train, test_data, y_test

def load_celeba(args, **kwargs):
    # set args
    args.input_size = [3, 64, 64]
    args.input_type = 'continuous'
    args.dynamic_binarization = False

    data_dir = '/hdd1/aiqingzhong/celeba/img_celeba.7z/celeba-hq/celeba-64/celeba-64/*.jpg'
    imgs = np.zeros((30000, 64, 64, 3))
    i = 0
    for imageFile in glob.glob(data_dir):
        img = np.array(Image.open(imageFile))
        imgs[i] = img
        i += 1
    imgs = imgs.transpose((0, 3, 1, 2))
    data = (imgs + 0.5) / 256.

    # shuffle data:
    np.random.shuffle(data)

    train_size = 26000
    val_size = 2000
    test_size = 2000

    x_train = data[0: train_size].reshape(-1, 3 * 64 * 64)
    x_val = data[train_size:(train_size + val_size)].reshape(-1, 3 * 64 * 64)
    x_test = data[(train_size + val_size):(train_size + val_size + test_size)].reshape(-1, 3 * 64 * 64)

    # idel label
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    indices = np.arange(len(x_train)).reshape(-1, 1)

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(indices),
                                     torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.prior == 'vampprior':
        if args.use_training_data_init == 1:
            args.pseudoinputs_std = 0.01
            init = x_train[0:args.number_components].T
            args.pseudoinputs_mean = torch.from_numpy(
                init + args.pseudoinputs_std * np.random.randn(np.prod(args.input_size), args.number_components)).float()
        else:
            args.pseudoinputs_mean = 0.05
            args.pseudoinputs_std = 0.01

    return train_loader, val_loader, test_loader, args


def load_dataset(args, training_num=None, use_fixed_validation=False, no_binarization=False, **kwargs):
    if training_num is not None:
        args.training_set_size = training_num

    if args.dataset_name == 'dynamic_mnist':
        if training_num is None:
            args.training_set_size = 50000
        args.input_size = [1, 28, 28]
        if args.continuous is True:
            print("*****Continuous Data*****")
            args.input_type = 'gray'
            args.dynamic_binarization = False
            no_binarization = True
        else:
            args.input_type = 'binary'
            args.dynamic_binarization = True

        train_loader, val_loader, test_loader, args = \
            dynamic_mnist_loader(args, use_fixed_validation, no_binarization=no_binarization).load_dataset(**kwargs)

    elif args.dataset_name == 'fashion_mnist':
        if training_num is None:
            args.training_set_size = 50000
        args.input_size = [1, 28, 28]
        if args.continuous is True:
            print("*****Continuous Data*****")
            args.input_type = 'gray'
            args.dynamic_binarization = False
            no_binarization = True
        else:
            args.input_type = 'binary'
            args.dynamic_binarization = True

        train_loader, val_loader, test_loader, args = \
            fashion_mnist_loader(args, use_fixed_validation, no_binarization=no_binarization).load_dataset(**kwargs)

    elif args.dataset_name == 'cifar10':
        args.training_set_size = 40000
        args.input_size = [3, 32, 32]
        args.input_type = 'continuous'
        train_loader, val_loader, test_loader, args = cifar10_loader(args).load_dataset(**kwargs)
    
    elif args.dataset_name == 'celeba':
        args.input_size = [3, 64, 64]
        args.input_type = 'continuous'
        train_loader, val_loader, test_loader, args = load_celeba(args, **kwargs)

    else:
        raise Exception('Wrong name of the dataset!')

    print('Train_size:', len(train_loader.dataset))
    if val_loader is not None:
        print('Val_size:', len(val_loader.dataset))
    if test_loader is not None:
        print('Test_size:', len(test_loader.dataset))
    return train_loader, val_loader, test_loader, args
