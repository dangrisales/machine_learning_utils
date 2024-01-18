#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:52:16 2024

@author: daniel
"""
import torch
import numpy as np

class BiModalGMU(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        """Initialize params."""
        super(BiModalGMU, self).__init__()
        self.input_size_1 = input_size[0]
        self.input_size_2 = input_size[1]
        self.hidden_size = hidden_size

        self.z1_weights = torch.nn.Linear(sum(input_size), hidden_size, bias=False)
        self.input1_weights = torch.nn.Linear(self.input_size_1, hidden_size, bias=False)
        self.input2_weights = torch.nn.Linear(self.input_size_2, hidden_size, bias=False)
        
    def forward(self, inputModalities):
        """Propogate input through the network."""
        
        x = torch.concat(inputModalities, axis=1)
        input1 = inputModalities[0]
        input2 = inputModalities[1]
        
        h_1 = self.input1_weights(input1)
        h_2 = self.input2_weights(input2)
        z = torch.sigmoid(self.z1_weights(x))
        
        h_1 = torch.tanh(h_1)
        h_2 = torch.tanh(h_2)

        h = z * h_1 + (1-z) * h_2
        return h, z
    

class GMUClassifier(torch.nn.Module):
    
    def __init__(self, modality_size_arr, weights_dimension, hidden_size, output_dim, dropoutProbability= 0.1):
        """Initialize params."""
        super(GMUClassifier, self).__init__()
        self.modality_size_arr = modality_size_arr
        self.weights_dim = weights_dimension
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        
        self.gmu_layer = BiModalGMU(modality_size_arr, weights_dimension)
        #self.bn1 = nn.BatchNorm1d(weights_dimension)
        self.fc1 = torch.nn.Linear(weights_dimension, hidden_size)
#        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.dropout = torch.nn.Dropout(dropoutProbability)
#        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fcOuput = torch.nn.Linear(hidden_size, output_dim)
        
    def forward(self, listModalities):
        """Propogate input through the network."""
        out, z = self.gmu_layer(listModalities)
        out = self.fc1(out)
        out = self.dropout(out)
        out = torch.relu(out)
        out = self.fcOuput(out)
        return out, z

#Example use GMU

model = BiModalGMU([768, 300], 200)

Modality1 = np.random.rand(1000, 768)
Modality2 = np.random.rand (1000, 300)
y_original = np.zeros(1000)
y_original[int(len(y_original)/2):] = 1

h, z = model([torch.FloatTensor(Modality1), torch.FloatTensor(Modality2)])


