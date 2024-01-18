#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:29:39 2023

@author: daniel
"""

import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

#%%
# def SoftmaxModified(x):
#   input_softmax = x.transpose(0,1)
#   function_activation = nn.Softmax(dim=1)
#   output = function_activation(input_softmax)
#   output = output.transpose(0,1)
#   return output


class SimpleClassifier_nn(torch.nn.Module):

    def __init__(self, input_size, hidden_size, dropoutProbability):
        """Initialize params."""
        super(SimpleClassifier_nn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.linear_layer= torch.nn.Linear(self.input_size, hidden_size)        
        self.output_layer = torch.nn.Linear(self.hidden_size, 3)
        self.dropout = torch.nn.Dropout(dropoutProbability)
        
    def forward(self, word_representation):
        """Propogate input through the network."""
        h1 = torch.relu(self.dropout(self.linear_layer(word_representation)))
        output = self.output_layer(h1)

        return output


class SimpleClassifier_nn_Dataset(Dataset):
    
    def __init__(self, x, y_data):
        self.x = x
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.x[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.x)