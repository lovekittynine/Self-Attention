#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:21:50 2019

@author: wsw
"""

# sa model train
import torch
from torch import nn,optim
import os
from self_attetion_model import SAModel
from dataset import make_dataloader
from tqdm import tqdm


def train():
  trainloader,testloader = make_dataloader()
  # build model
  model = SAModel()
  # loss func
  loss_func = nn.CrossEntropyLoss()
  # optimzier
  optimizier = optim.Adam(model.parameters(),lr=1e-3)
  # configuration
  epochs = 10
  
  # training
  for epoch in range(epochs):
    model.train()
    pbar = tqdm(trainloader)
    for image,label in pbar:
      # forward
      output = model(image)
      # compute loss
      loss = loss_func(output,label)
      optimizier.zero_grad()
      loss.backward()
      optimizier.step()
      # compute batch accuracy
      predicts = torch.argmax(output,dim=-1)
      accu = torch.sum(predicts==label).float()/image.size(0)
      pbar.set_description('Epoch:[{:02d}]-Loss:{:.3f}-Accu:{:.3f}'\
                           .format(epoch+1,loss.item(),accu.item()))
    # testing
    model.eval()
    with torch.no_grad():
      corrects = 0
      total_nums = 0
      for image,label in tqdm(testloader):
        output = model(image)
        predicts = torch.argmax(output,dim=-1)
        corrects += (predicts==label).sum()
        total_nums += label.size(0)
      test_accu = corrects.float()/total_nums
      print('Epoch:[{:02d}]-Test_Accu:{:.3f}'.format(epoch+1,test_accu.item()))
      
if __name__ == '__main__':
  train()