#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:12:35 2020

@author: daniel
"""

import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sb

def graphic_resultsROC(y_real, y_scores, color_roc, titleAUC = 'AUC: ', newFigure = True, title = 'ROC curve' ):
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
    plt.plot(fpr, tpr, color = color_roc, label = titleAUC+ '%.2f' % AUC  )
    plt.legend(loc='best', title='AUC')


def make_radar_chart(name, stats, attribute_labels, label_model, pathResults):
    
    parameters = {'xtick.labelsize': 17, 'ytick.labelsize': 17}
                  
    plt.rcParams.update(parameters)


    labels = np.array(attribute_labels)

    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))


    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure(figsize=(10,10))
    
    ax = fig.add_subplot(111, polar=True)
    
    ax.plot(angles, stats, 'o-', linewidth=2, c='b', label= label_model)
    ax.fill(angles, stats, alpha=0.25, c='b')
    
#     ax.plot(angles, stats2, 'o-', linewidth=2, c='r', label= 'After')
#     ax.fill(angles, stats2, alpha=0.25, c='r')

    ax.set_thetagrids(angles[:-1 ] * 180/np.pi, [])
    ax.set_ylim((0,100))
    ax.set_yticklabels([20,40,60,80,100])
    ax.set_xticklabels(labels)
    ax.set_title(name, fontsize=20)
    ax.grid(True)
    ax.legend(loc=3,fontsize=16)

    fig.savefig(pathResults)

    plt.show()


def make_radar_chart_2_models(name, stats, stats2, attribute_labels, label_model, pathResults, title):
    
    parameters = {'xtick.labelsize': 17, 'ytick.labelsize': 17}
                  
    plt.rcParams.update(parameters)


    labels = np.array(attribute_labels)

    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    stats2 = np.concatenate((stats2,[stats2[0]]))


    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure(figsize=(10,10))
    
    ax = fig.add_subplot(111, polar=True)
    
    ax.plot(angles, stats, 'o-', linewidth=2, c='b', label= label_model[0])
    ax.fill(angles, stats, alpha=0.25, c='b')
    
    ax.plot(angles, stats2, 'o-', linewidth=2, c='orange', label= label_model[1])
    ax.fill(angles, stats2, alpha=0.25, c='orange')
    
    

    ax.set_thetagrids(angles[:-1 ] * 180/np.pi, [])
    #ax.set_ylim((0, max(angles)))
    #ax.set_yticklabels(np.linspace(0, max(angles), len(angles)))
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=20)
    ax.grid(True)
    ax.legend(loc=3,fontsize=16)

    fig.savefig(pathResults)

    plt.show()


    
# filePath = '/home/daniel/Escritorio/GITA/TSDNewVersion/BERT/resultsProteccion.csv'
# results = pd.read_csv(filePath, header = None)

# metric = np.asarray(results[1])

# index_real = np.where(metric == 'y_real')[0]
# index_score = np.where(metric == 'score')[0]
# index_pred = np.where(metric == 'y_pred')[0]


# y_real_for_experiment = [np.asarray(results.iloc[i,2:]) for i in index_real]
# y_pred_for_experiment = [np.asarray(results.iloc[i,2:]) for i in index_pred]
# score_for_experiment = [np.asarray(results.iloc[i,2:]) for i in index_score]

# y_real_c = [float(value) for value in np.hstack(y_real_for_experiment)]
# y_pred_c = [float(value) for value in np.hstack(y_pred_for_experiment)]
# score_c = [float(value) for value in np.hstack(score_for_experiment)]

# grafic_results(y_real_c, score_c, y_pred_c)
