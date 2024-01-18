#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:40:45 2020

@author: daniel
"""
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
def plot_cm(y_true, y_pred, labels, figsize=(15,15), num_class = 2):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Real'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
  #  sns.set(font_scale=1.6)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, vmin=0, vmax=len(y_true)/len(set(y_true)), annot_kws={"fontsize":30}, cmap='Blues')
    plt.yticks([i + 0.5 for i in range(0,num_class)],labels, rotation='vertical',fontsize=5)
    plt.xticks([i + 0.5 for i in range(0,num_class)],labels, rotation='horizontal',fontsize=5)
    
    plt.tick_params(labelsize=30)
    plt.xlabel('Predicted',fontsize=40)
    plt.ylabel('Real',fontsize=40)
    
# y_true = [1,1,2,2,3,3,4,4,5,5,6,6,7,7]
# y_pred = y_true
    
# plot_cm(y_true, y_pred)