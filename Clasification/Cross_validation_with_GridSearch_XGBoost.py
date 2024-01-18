#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:53:32 2020

@author: daniel
"""
import numpy as np
from sklearn import preprocessing, svm, model_selection,metrics
import time
from tqdm import tqdm
import sys
import pandas as pd
import csv
from scipy import stats
import os
from xgboost import XGBClassifier


def save_results(file_csv, name_result, fold, result):
    #file_csv: nombre.csv del archivo donde quiero guardar el resultado,
    # ejm 'results_spanish.csv'
    #name_result: nombre del resultado a evaluar ejm: loss_train
    #fold: K del fold en el que se llama la funcion ejm: 3
    #result: list a guardar ejm: loss_aux o [40.0]
    result_aux=result
    result_aux.insert(0,name_result)
    result_aux.insert(0,fold)
    with open(file_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(result_aux)

def balanceDataBase(Features, Labels):
   
   max_class = min([len(np.where(Labels ==1)[0]), len(np.where(Labels ==0)[0])])
   indexNegativeClass = np.where(Labels == 0)[0][0:max_class]
   indexPositiveClass = np.where(Labels == 1)[0][0:max_class]
   
   indexF = np.hstack((indexNegativeClass, indexPositiveClass))
   
   Features_Balance = Features[indexF]

   
   Labels = Labels[indexF]
   
   
   return Features_Balance, Labels

def Cross_validation_XGboost(X,y,folds,name_csv,
                             iterations=10,number_folds_validation=5,
                             n_job = -1, shuffle_bool=True):
    
#https://arxiv.org/ftp/arxiv/papers/1901/1901.08433.pdf parameters of tuning XGBoost
    
    parameters_xgboost = {
        'max_depth': np.arange(1,8,2),
        'min_child_weight': np.arange(1,6,2),
        'gamma': [i/10.0 for i in range(0,5)],
        'learning_rate': np.arange(0.01, 0.1, 0.01),
        'subsample':[i/100.0 for i in range(50,90,10)],
        'colsample_bytree':[i/100.0 for i in range(50,90,10)]}
    



    if os.path.exists(name_csv):
        os.remove(name_csv)
    
    accuracy_experiments = []
    sensibility_experiments = []
    specificity_experiments = []
    f1score_experiments = []
    best_C_experiments = []
    best_G_experiments = []
    auc_experiments = []
    parameter_experiments = []
    
    for index in range(iterations):
        index = index + 1
        score_array = []
        y_pred_array = []
        y_real_array = []
        best_C_array = []
        best_G_array = []
        parameters_array = []
  
        print('# iteraccion: '+str(index))
        with tqdm(total = 100, file = sys.stdout) as pbar:
            step = 100/folds
            kf = model_selection.KFold(n_splits=folds,shuffle=shuffle_bool)
            for train_index, test_index in kf.split(X):
                
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_train, y_train = balanceDataBase(X_train, y_train)
                
                normalize_transfor=preprocessing.StandardScaler().fit(X_train)
                X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
                            
  
                xgboost_clf = XGBClassifier()
                clf = model_selection.GridSearchCV(xgboost_clf, parameters_xgboost, cv=number_folds_validation, n_jobs = n_job)
                clf.fit(X_train,y_train)                
                
                #Best parameters and testSet
                
                clf_final = XGBClassifier(max_depth= clf.best_estimator_.max_depth,
                                          learning_rate = clf.best_estimator_.learning_rate,
                                          min_child_weight = clf.best_estimator_.min_child_weight,
                                          subsample = clf.best_estimator_.subsample,
                                          gamma = clf.best_estimator_.gamma,
                                          colsample_bytree = clf.best_estimator_.colsample_bytree)
        
    
                clf_final.fit(X_train_N,y_train)
                y_pred = clf_final.predict(X_test_N)
                parameters = [ clf.best_estimator_.max_depth,
                               clf.best_estimator_.learning_rate,
                               clf.best_estimator_.min_child_weight,
                               clf.best_estimator_.subsample,
                               clf.best_estimator_.gamma,
                               clf.best_estimator_.colsample_bytree]
                
                # Results in each fold
                parameters_array.append(parameters)
                y_real_array.append(y_test)
                y_pred_array.append(y_pred)
                pbar.update(step)
            

        y_real_total = np.hstack(y_real_array)
        y_pred_total = np.hstack(y_pred_array)
        tn, fp, fn, tp = metrics.confusion_matrix(y_real_total, y_pred_total).ravel()
        
        acc_experiment = metrics.accuracy_score(y_real_total, y_pred_total)*100
        sen_experiment = tp/(tp+fn)*100
        spe_experiment = tn/(tn+fp)*100
        f1_experiment = metrics.f1_score(y_real_total,y_pred_total)*100
        
        
        accuracy_experiments.append(acc_experiment)
        sensibility_experiments.append(sen_experiment)
        specificity_experiments.append(spe_experiment)
        f1score_experiments.append(f1_experiment)
        parameter_experiments.append([stats.mode(np.asarray(parameters_array)[:,parameter])[0][0] for parameter in range(0,6)])

        
        save_results(name_csv, 'y_real', index, list(y_real_total))
        save_results(name_csv, 'y_pred', index, list(y_pred_total))
        save_results(name_csv, 'Accuracy', index, [acc_experiment])
        save_results(name_csv, 'Sensitivity', index, [sen_experiment])
        save_results(name_csv, 'Specificity', index, [spe_experiment])
        save_results(name_csv, 'F1Score', index, [f1_experiment])
        save_results(name_csv, 'parameters', index, parameters)
    
    save_results(name_csv, ' ', ' ', [' '])        
    save_results(name_csv, ' ', ' ', ['Mean', 'Std'])
    save_results(name_csv, 'Accuracy', 'Final', [np.mean(accuracy_experiments), np.std(accuracy_experiments)])
    save_results(name_csv, 'Sensitivity', 'Final', [np.mean(sensibility_experiments), np.std(sensibility_experiments)])
    save_results(name_csv, 'Specificity', 'Final', [np.mean(specificity_experiments), np.std(specificity_experiments)])
    save_results(name_csv, 'F1Score', 'Final', [np.mean(f1score_experiments), np.std(f1score_experiments)])    
    save_results(name_csv, 'Parameters Mode', 'Final', [stats.mode(np.asarray(parameter_experiments)[:,0])[0][0], 
                                                        stats.mode(np.asarray(parameter_experiments)[:,1])[0][0], 
                                                        stats.mode(np.asarray(parameter_experiments)[:,2])[0][0],
                                                        stats.mode(np.asarray(parameter_experiments)[:,3])[0][0],
                                                        stats.mode(np.asarray(parameter_experiments)[:,4])[0][0],
                                                        stats.mode(np.asarray(parameter_experiments)[:,5])[0][0]])

    save_results(name_csv, ' ', ' ', [' '])
    
Feat_GloVe = np.loadtxt('Features-GloVe-300.txt')[0:20, 0:5]
#Load labels
path_base='LabelsClassification.csv'
labels_data = pd.read_csv(path_base,delimiter=',')
y = list(labels_data['Agr'])[0:20] 
y = np.asarray(y)

Cross_validation_XGboost(Feat_GloVe,y,10,'results.csv',iterations=5,number_folds_validation=10)

 