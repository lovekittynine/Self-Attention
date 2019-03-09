#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:04:02 2019

@author: wsw
"""

# make dataset
from torchvision.datasets import FashionMNIST
from torch.utils import data
from torchvision import transforms

def make_dataset():
  data_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.5],[0.5])])
  train_dataset = FashionMNIST(root='./fashion_data',
                               train=True,
                               transform=data_transform,
                               download=True)
  test_dataset = FashionMNIST(root='./fashion_data',
                              train=False,
                              transform=data_transform,
                              download=True)
  return train_dataset,test_dataset

def make_dataloader():
  train_dataset,test_dataset = make_dataset()
  train_dataloader = data.DataLoader(train_dataset,
                                     batch_size=256,
                                     shuffle=True,
                                     num_workers=8)
  test_dataloader = data.DataLoader(test_dataset,
                                    batch_size=256,
                                    num_workers=8)
  return train_dataloader,test_dataloader


if __name__ == '__main__':
  trainloader,testloader = make_dataloader()