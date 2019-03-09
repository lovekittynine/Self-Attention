#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:47:36 2019

@author: wsw
"""

# self-attention model
import torch
from torch import nn


class SAModel(nn.Module):
  
  def __init__(self,num_class=10):
    super(SAModel,self).__init__()
    self.features = nn.Sequential(
                                  # output->14x14x16
                                  nn.Conv2d(1,16,3,padding=1,bias=False),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2,2),
                                  # add self attention
                                  SelfAttention(16),
                                  # output->7x7x32
                                  nn.Conv2d(16,32,3,padding=1,bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2,2),
                                  # add self attention
                                  SelfAttention(32),
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


class SelfAttention(nn.Module):
  
  def __init__(self,inplanes):
    super(SelfAttention,self).__init__()
    self.inplanes = inplanes
    # theta transform
    self.theta = nn.Conv2d(self.inplanes,self.inplanes//2,1)
    # phi transform
    self.phi = nn.Sequential(nn.Conv2d(self.inplanes,self.inplanes//2,1),
                             # nn.MaxPool2d(2,2)
                             )
    # function g
    self.g_func = nn.Sequential(nn.Conv2d(self.inplanes,self.inplanes//2,1),
                                # nn.MaxPool2d(2,2)
                                )
    
    # w transform to match the channel
    self.w = nn.Conv2d(self.inplanes//2,self.inplanes,1)
  
  def forward(self,xs):
    
    theta = self.theta(xs)
    N,C,W,H = theta.size()
    theta = theta.view(N,C,H*W).transpose(2,1)
    # print(theta.shape)
    phi = self.phi(xs)
    phi = phi.view(N,C,-1)
    # compute attention
    attention = theta.bmm(phi)
    assert attention.size()==(N,H*W,H*W)
    attention = nn.functional.softmax(attention,dim=-1)
    # g transform
    g = self.g_func(xs)
    g = g.view(N,C,-1)
    # final response
    response = g.bmm(attention.transpose(2,1))
    response = response.view(N,C,W,H)
    # matching channel
    response = self.w(response)
    output = response + xs
    return output
    

if __name__ == '__main__':
  xs = torch.randn(size=(128,1,28,28))
  sa_model = SAModel()
  output = sa_model(xs)
  print(output.size())