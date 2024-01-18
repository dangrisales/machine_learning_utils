# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:26:23 2020

@author: Felipe
"""
import numpy as np
from sklearn import preprocessing, svm, model_selection,metrics
import time
#from tqdm import tqdm
import sys
import pandas as pd
import csv
from scipy import stats
import os
from sklearn.ensemble import RandomForestClassifier

#%%
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

def Cross_validation_RF(X,y,folds,name_csv,iterations=10,number_folds_validation=5,
                         balance=True,
                         n_job = -1, shuffle_bool=True):
    
    if os.path.exists(name_csv):
        os.remove(name_csv)
    
    parameters_rf = {
        'n_estimators': np.arange(100,400,50),
        'max_features': [x for x in np.arange(0.1,1,0.3)]+[int(np.sqrt(np.shape(X)[1]))]
        }
       
    
    accuracy_experiments = []
    sensibility_experiments = []
    specificity_experiments = []
    f1score_experiments = []
    best_n_estimators_experiments = []
    best_max_features_experiments = []
    auc_experiments = []
    
    if(balance):
        X, y = balanceDataBase(X, y)
    samples = len(y)
    
    for index in range(iterations):
        index = index + 1
        y_pred_array = []
        y_real_array = []
        score_array = []
        best_n_est_array = []
        best_max_feat_array = []
  
        print('# iteracion: '+str(index))
        with tqdm(total = 100, file = sys.stdout) as pbar:
            step = 100/folds
            kf = model_selection.KFold(n_splits=folds,shuffle=shuffle_bool)
            #cont = 1
            for train_index, test_index in kf.split(X):
                
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                
                normalize_transfor=preprocessing.StandardScaler().fit(X_train)
                X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
                
                rfc = RandomForestClassifier(n_jobs=n_job)
                clf = model_selection.GridSearchCV(rfc, parameters_rf, cv=number_folds_validation, n_jobs = n_job)
                clf.fit(X_train,y_train)
                
                best_n_est = clf.best_estimator_.n_estimators
                best_m_feat = clf.best_estimator_.max_features
                
                #Best parameters and testSet
                
                RF_Classifier = RandomForestClassifier(n_estimators=best_n_est, max_features = best_m_feat, n_jobs=n_job)
                RF_Classifier.fit(X_train_N,y_train)
                y_pred=RF_Classifier.predict(X_test_N)
                score_test = RF_Classifier.predict_proba(X_test_N)[:, 1]
                # Results in each fold
                
                y_real_array.append(y_test)
                y_pred_array.append(y_pred)
                score_array.append(score_test)
                best_n_est_array.append(best_n_est)
                best_max_feat_array.append(best_m_feat)
                
                #print("Fold: ",cont)
                #cont = cont +1
                pbar.update(step)
        
        score_pred_total = np.hstack(score_array)
        y_real_total = np.hstack(y_real_array)
        y_pred_total = np.hstack(y_pred_array)
        tn, fp, fn, tp = metrics.confusion_matrix(y_real_total, y_pred_total).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_real_total,score_pred_total)
        
        acc_experiment = metrics.accuracy_score(y_real_total, y_pred_total)*100
        sen_experiment = tp/(tp+fn)*100
        spe_experiment = tn/(tn+fp)*100
        f1_experiment = metrics.f1_score(y_real_total,y_pred_total)*100
        n_est_experiment = stats.mode(best_n_est_array)[0][0]
        max_feat_experiment = stats.mode(best_max_feat_array)[0][0]
        auc_experiment = metrics.auc(fpr,tpr)
        
        
        accuracy_experiments.append(acc_experiment)
        sensibility_experiments.append(sen_experiment)
        specificity_experiments.append(spe_experiment)
        f1score_experiments.append(f1_experiment)
        auc_experiments.append(auc_experiment)
        best_n_estimators_experiments.append(n_est_experiment)
        best_max_features_experiments.append(max_feat_experiment)
        
        save_results(name_csv, 'y_real', index, list(y_real_total))
        save_results(name_csv, 'y_pred', index, list(y_pred_total))
        save_results(name_csv, 'score', index, list(score_pred_total))
        save_results(name_csv, 'Accuracy', index, [acc_experiment])
        save_results(name_csv, 'Sensitivity', index, [sen_experiment])
        save_results(name_csv, 'Specificity', index, [spe_experiment])
        save_results(name_csv, 'AUC', index, [auc_experiment])
        save_results(name_csv, 'F1Score', index, [f1_experiment])
        save_results(name_csv, 'N est. and Max feat.', index, [n_est_experiment, max_feat_experiment])
    
    save_results(name_csv, ' ', ' ', [' '])        
    save_results(name_csv, ' ', ' ', ['Mean', 'Std'])
    save_results(name_csv, 'Accuracy', 'Final', [np.mean(accuracy_experiments), np.std(accuracy_experiments)])
    save_results(name_csv, 'Sensitivity', 'Final', [np.mean(sensibility_experiments), np.std(sensibility_experiments)])
    save_results(name_csv, 'Specificity', 'Final', [np.mean(specificity_experiments), np.std(specificity_experiments)])
    save_results(name_csv, 'F1Score', 'Final', [np.mean(f1score_experiments), np.std(f1score_experiments)])    
    save_results(name_csv, 'AUC', 'Final', [np.mean(auc_experiments), np.std(auc_experiments)])    
    save_results(name_csv, 'N estimators', 'Final', [np.mean(best_n_estimators_experiments), np.std(best_n_estimators_experiments)])
    save_results(name_csv, 'Max features', 'Final', [np.mean(best_max_features_experiments), np.std(best_max_features_experiments)])
    save_results(name_csv, ' ', ' ', [' '])
    save_results(name_csv, ' ', ' ', ['N est', 'Max feat'])
    save_results(name_csv, 'Parameters Mode', 'Final', [stats.mode(best_n_estimators_experiments)[0][0], stats.mode(best_max_features_experiments)[0][0]])
    save_results(name_csv, '# Samples used', ' ', [samples])

Feat_GloVe = np.loadtxt('Features-GloVe-300.txt')[0:10,0:20]
#Load labels
path_base='LabelsClassification.csv'
labels_data = pd.read_csv(path_base,delimiter=',')
y = list(labels_data['Agr'])
y = np.asarray(y)[0:10]

Cross_validation_RF(Feat_GloVe,y,2,'results.csv',iterations=2,number_folds_validation=2)
