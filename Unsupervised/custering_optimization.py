#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:15:33 2023

@author: daniel
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import  metrics
#import pandas as pd
#from sklearn.cluster import KMeans
import skfuzzy as fuzz
#from statistics import mode
#import seaborn as sns


#%%
def find_best_clusters(val_scores, cluster_list, metrics_interpretation = {'Silhouette': 1, 'CH_score':1, 'DB_score':0}):
    #metrics_interpretation 1 if high score is best 0 in otherwise
    #sort based on 
    best_cluster_list = []
    for key_metric in val_scores.keys():
        if metrics_interpretation[key_metric] == 1:
            best_cluster_metric = cluster_list[np.argmax(val_scores[key_metric])]                        
        elif metrics_interpretation[key_metric] == 0:
            best_cluster_metric = cluster_list[np.argmin(val_scores[key_metric])]
        best_cluster_list.append(best_cluster_metric)
    
    best_cluster = np.mean(best_cluster_list)
    return best_cluster_list, int(best_cluster)
        
    

def cluster_definition_fuzzy_cmeans(clusters_list, x, pathResult):
    
    validation_scores = {}
    validation_scores['Silhouette'] = []
    validation_scores['DB_score'] = []
    validation_scores['CH_score'] = []
    
    for cluster in clusters_list:
        m = 2 #fuzzy parameter
        results_cluster = fuzz.cluster.cmeans(x.T, cluster, m, error=0.005, maxiter=1000, init=None)

        
        centers = results_cluster[0]
        labels = np.argmax(results_cluster[1], axis=0)
        # kmeans = KMeans(n_clusters=cluster, random_state=0).fit(x)
        # labels = kmeans.predict(x)

        #plt.scatter(x[-6,0], x[-6,1], color='r')
        
        validation_scores['Silhouette'].append( metrics.silhouette_score(x, labels))
        validation_scores['DB_score'].append(metrics.davies_bouldin_score(x, labels))
        validation_scores['CH_score'].append(metrics.calinski_harabasz_score(x, labels))
    
    

    
    plt.figure()
    plt.plot(clusters_list, validation_scores['Silhouette'], 'o-', label = 'Silhouette')
    plt.plot(clusters_list, validation_scores['DB_score']/max(validation_scores['DB_score']),'o-', label = 'DB_score')
    plt.plot(clusters_list, validation_scores['CH_score']/max(validation_scores['CH_score']), 'o-',  label = 'CH_score')
    plt.legend()
    plt.savefig(pathResult)
    
    return validation_scores

def fuzzy_c_means_implementation(X, cluster_number):
    m = 2 #fuzzy parameter
    results_cluster = fuzz.cluster.cmeans(X.T, cluster_number, m, error=0.005, maxiter=1000, init=None)

    
    centers = results_cluster[0]
    probabilities = results_cluster[1]
    labels = np.argmax(results_cluster[1], axis=0)  
    
    return centers, probabilities, labels


# x1 = np.random.normal(3,0.01, (100,3))
# x2 = np.random.normal(0,0.01, (100,3))
# x3 = np.random.normal(2,0.01, (100,3))
# x4 = np.random.normal(3.2,0.01, (100,3))
# x = np.concatenate((x1, x2, x3, x4))
# cluster_list = list(range(2,11))
# val_scores = cluster_definition_fuzzy_cmeans(cluster_list, x)

# best_cluster_metric, best_cluster = find_best_clusters(val_scores, cluster_list)

