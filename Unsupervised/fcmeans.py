#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:53:54 2023

@author: daniel
"""

import fcmeans as f_cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  metrics 
import pandas as pd



#%%
def cluster_definition(cluster_list, x):
    
    validation_scores = {}
    validation_scores['Silhouette'] = []
    validation_scores['DB_score'] = []
    validation_scores['CH_score'] = []
    
    for cluster in clusters_list:
        fuzzy_model = f_cm.FCM(n_clusters=cluster)
        fuzzy_model.fit(x)
        
        centers = fuzzy_model.centers
        labels = fuzzy_model.predict(x) 

        #plt.scatter(x[-6,0], x[-6,1], color='r')
        
        validation_scores['Silhouette'].append( metrics.silhouette_score(x, labels))
        validation_scores['DB_score'].append(metrics.davies_bouldin_score(x, labels))
        validation_scores['CH_score'].append(metrics.calinski_harabasz_score(x, labels))
    
    
    best_cluster_list = [clusters_list[np.argmax(validation_scores['Silhouette'])], 
                    clusters_list[np.argmin(validation_scores['DB_score'])],
                    clusters_list[np.argmax(validation_scores['CH_score'])]]
        
    
    best_cluster = np.mean(best_cluster_list)
    
    plt.figure()
    plt.plot(clusters_list, validation_scores['Silhouette'], 'o-', label = 'Silhouette')
    plt.plot(clusters_list, validation_scores['DB_score'],'o-', label = 'DB_score')
    plt.plot(clusters_list, validation_scores['CH_score']/max(validation_scores['CH_score']), 'o-',  label = 'CH_score')
    plt.legend()
    
    return best_cluster, validation_scores, best_cluster_list
    
#%%    

database = pd.read_excel('../Database/todos_pd_promedio_features_psico.xlsx')

#%%    
clusters_list = list(np.arange(2,15,1))    
x1 = np.random.normal(-0.5, 0.1, size=(100,2))
x2 = np.random.normal(0, 0.1, size=(100,2))
x3 = np.random.normal(0.5, 0.1, size=(100,2))
x4 = np.random.normal(1, 0.1, size=(100,2))

x = np.vstack((x1,x2,x3))

cluster_definition(clusters_list, x)