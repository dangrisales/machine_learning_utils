#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 10:34:38 2022

@author: daniel
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def graphicDistributions(scores_o, y_labels_o, bins, positiveClass = 'Positive', negativeClass = 'Negative', colorNegative= 'White', colorPositive = 'Black'):
    plt.figure(figsize=(8,6))
    sns.distplot(scores_o[y_labels_o==0], label=negativeClass,
                 hist_kws={'edgecolor':'black','color':colorNegative},
                 kde_kws={"color": "k", "linestyle":'--'}, bins = bins)
    sns.distplot(scores_o[y_labels_o==1], label=positiveClass, color=colorPositive,
                 hist_kws={'edgecolor':'black','color':colorPositive},
                 kde_kws={"color": "k"}, bins = bins)
    plt.xlabel('Decision scores',fontsize=22)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=20)
    plt.savefig("distribution.pdf")
    plt.figure(figsize=(8,6))

    
scores = np.asarray([0.1, 0, 0.3, 0.4, 0.5, 0, 0, 0, 0, 0.5, 0.6, 0.7,1,1,1,1,0.7,0.6])
label = np.asarray([0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1])
graphicDistributions(scores, label, bins = 4)
