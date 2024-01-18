#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:02:01 2023

@author: daniel
"""


import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def grafic_resultsROC(y_real, y_scores, color_roc = 'blue', titleAUC = 'AUC: ', newFigure = True, title = 'ROC curve' ):
    if newFigure:
        fig = plt.figure(figsize = (15,15))
        ax = fig.add_subplot(111)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate', fontsize=45)
        plt.ylabel('True positive rate', fontsize=45)
        plt.title(title, fontsize=45)
        parameters = {
                      'legend.fontsize': 40,
                      'lines.linewidth': 7,
                      'legend.title_fontsize':45}
        plt.rcParams.update(parameters)
        ax.set_aspect('equal', adjustable='box')
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)
        


    fpr, tpr, thresholds = roc_curve(y_real, y_scores)
    AUC = auc(fpr, tpr)
    #plt.figure(figsize = (10,10))
    plt.plot(fpr, tpr, color=color_roc, label = titleAUC+ '%.2f' % AUC  )
    plt.legend(loc='best', title='AUC')