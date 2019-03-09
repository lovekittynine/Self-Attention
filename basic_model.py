#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:48:34 2019

@author: wsw
"""

# build basic model
import torch
from torch import nn


class BasicModel(nn.Module):
  
  def __init__(self,num_class=10):
    super(BasicModel,self).__init__()
    self.features = nn.Sequential(
                                  # output->14x14x16
                                  nn.Conv2d(1,16,3,padding=1,bias=False),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2,2),
                                  # output->7x7x32
                                  nn.Conv2d(16,32,3,padding=1,bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2,2),
                                  # output->4x4x64
                                  nn.Conv2d(32,64,3,padding=1,bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2,2,padding=1),
                                  # output->1x1x128
                                  nn.Conv2d(64,128,4,bias=False),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(True)
                                  )
    self.classifer = nn.Linear(128,num_class)
    
  def forward(self,xs):
    bs = xs.size(0)
    xs = self.features(xs)
    xs = xs.view(bs,-1)
    xs = self.classifer(xs)
    return xs
  

if __name__ == '__main__':
  xs = torch.randn(size=(128,1,28,28))
  basic_model = BasicModel()
  output = basic_model(xs)
  print(output.size())