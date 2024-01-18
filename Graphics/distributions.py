#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:27:57 2019

@author: luisparra
"""


import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def graphicDistributions(scores_o, y_labels_o, positiveClass = 'Positive', negativeClass = 'Negative', colorNegative= 'White', colorPositive = 'Black', ylim = [0,2], filename='distribution-opensmile+dur.pdf'):
    plt.figure(figsize=(8,6))
    sns.distplot(scores_o[y_labels_o==0], label=negativeClass,
                 hist_kws={'edgecolor':'black','color':colorNegative},
                 kde_kws={"color": colorNegative, "linestyle":'--'})
    sns.distplot(scores_o[y_labels_o==1], label=positiveClass, color=colorPositive,
                 hist_kws={'edgecolor':'black','color':colorPositive},
                 kde_kws={"color": colorPositive})
    plt.xlabel('Decision scores',fontsize=22)
    plt.ylabel('Density',fontsize=22)
    plt.ylim(ylim)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=20)
    plt.savefig(filename)

def graphicDistributionsSpecial(df, key1, key2, pal, title, y_limits, filename='distribution-opensmile+dur.pdf'):
    #plt.figure(figsize=[15,7])
    fig, ax = plt.subplots(figsize=(100,7))
    
    # sns.displot(scores_o[y_labels_o==0], label=negativeClass,
    #              kde_kws={"color": colorPositive, "linestyle":'--'})
    # sns.displot(scores_o[y_labels_o==1], label=positiveClass, color=colorPositive,
    #              kde_kws={"color": colorPositive})
    sns.displot( data=df, x=key1, hue=key2, kind="kde", palette=pal, ax = ax)
    #sns.displot(data=df, x=key1, hue=key2, multiple="stack")
    plt.xlabel('Decision scores',fontsize=15)
    plt.ylabel('Density',fontsize=15)
    #plt.title(title, fontsize=15)
    plt.ylim(y_limits)
    #plt.legend(fontsize=15)
    plt.tick_params(labelsize=10)
    #plt.show()
    plt.savefig(filename)

def boxplot(scores_o, y_labels_o, positiveClass = 'Positive', negativeClass = 'Negative', xlim = [-1, 1], ylim = [-1, 1], pal = 'Greys'):
    plt.figure(figsize=(10,3))
    
    df = pd.DataFrame()
    df['scores'] = scores_o
    y_labels_categorical = [positiveClass if l == 1 else negativeClass for l in y_labels_o]
    df['class'] = y_labels_categorical
    sns.boxplot(x= 'scores', y = 'class', data=df, palette = pal)
    plt.xlabel('Decision scores',fontsize=22)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=20)
    plt.xlim(xlim)
    #plt.ylim(ylim)

def graphicFeaturesPCA(feature_matrix, colors, key_colors):
    plt.figure(figsize=(7,7))
    pca_reduction = PCA(2)
    X_red = pca_reduction.fit_transform(feature_matrix)
    plt.scatter(X_red[:,0], X_red[:,1], c=colors, cmap='rocket')
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    plt.colorbar(label='Color ' + key_colors)
    plt.xlabel('PC1', fontsize = 20)
    plt.ylabel('PC2', fontsize = 20)
    
# scores = np.asarray([0.1, 0, 0.3, 0.4, 0.5, 0, 0, 0, 0, 0.5, 0.6, 0.7,1,1,1,1,0.7,0.6])
# label = np.asarray([0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1])
# graphicDistributions(scores, label)
