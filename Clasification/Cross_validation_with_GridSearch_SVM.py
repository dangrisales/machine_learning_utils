#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:27:17 2019

@author: gita
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

def balanceDataBaseMulticlass(Features, Labels):
   
    
   max_class = min([len(np.where(np.asarray(Labels) == i)[0]) for i in set(Labels)])


   indexClass = [np.where(np.asarray(Labels) == n)[0][0:max_class] for n in set(Labels)]

   
   indexF = np.hstack(indexClass)
   
   Features_Balance = np.asarray(Features)[indexF]

   
   Labels = np.asarray(Labels)[indexF]
   
   
   return Features_Balance, Labels


#%%

def Cross_validation_SVM(X,y,folds,name_csv,iterations=10,number_folds_validation=5,
                         max_c=100,min_c=0.001,number_c=6,max_gamma=10000,
                         min_gamma=0.00001,number_gamma=10,kernel_linear=False,
                         n_job = -1, shuffle_bool=True, balance=True):
    
    if os.path.exists(name_csv):
        os.remove(name_csv)
    
    
    C_array =np.geomspace(min_c,max_c,number_c)
    G_array=np.geomspace(min_gamma,max_gamma,number_gamma)
    
    if balance:
        X, y = balanceDataBase(X, y)
                
    samples = len(y)
    
    
    accuracy_experiments = []
    sensibility_experiments = []
    specificity_experiments = []
    f1score_experiments = []
    best_C_experiments = []
    best_G_experiments = []
    auc_experiments = []
    
    for index in range(iterations):
        index = index + 1
        score_array = []
        y_pred_array = []
        y_real_array = []
        best_C_array = []
        best_G_array = []

        print('# iteraccion: '+str(index))
        with tqdm(total = 100, file = sys.stdout) as pbar:
            step = 100/folds
            kf = model_selection.KFold(n_splits=folds,shuffle=shuffle_bool)
            for train_index, test_index in kf.split(X):
                
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                
                normalize_transfor=preprocessing.StandardScaler().fit(X_train)
                X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
                
                if kernel_linear:
                    parameters = {'kernel':['linear','rbf'], 'C':C_array}
                    svc = svm.SVC(gamma = 'scale')
                    clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
                    clf.fit(X_train,y_train)                
                    best_C=clf.best_estimator_.C
                    best_G='scale'
                
                else:
                    parameters = {'kernel':['rbf'], 'C':C_array,'gamma':G_array }
                    svc = svm.SVC()
                    clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
                    clf.fit(X_train,y_train)                
                    best_C=clf.best_estimator_.C
                    best_G=clf.best_estimator_.gamma
                
                #Best parameters and testSet
                
                SVM_Classifier = svm.SVC(kernel = clf.best_estimator_.kernel , gamma=best_G, C=best_C)
                SVM_Classifier.fit(X_train_N,y_train)
                y_pred=SVM_Classifier.predict(X_test_N)
                score_test = SVM_Classifier.decision_function(X_test_N)
                
                # Results in each fold
                
                y_real_array.append(y_test)
                y_pred_array.append(y_pred)
                score_array.append(score_test)
                best_C_array.append(best_C)
                best_G_array.append(best_G)
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
        C_experiment = stats.mode(best_C_array)[0][0]
        auc_experiment = metrics.auc(fpr,tpr)
        
        
        accuracy_experiments.append(acc_experiment)
        sensibility_experiments.append(sen_experiment)
        specificity_experiments.append(spe_experiment)
        f1score_experiments.append(f1_experiment)
        auc_experiments.append(auc_experiment)
        best_C_experiments.append(C_experiment)

        
        save_results(name_csv, 'y_real', index, list(y_real_total))
        save_results(name_csv, 'y_pred', index, list(y_pred_total))
        save_results(name_csv, 'score', index, list(score_pred_total))
        save_results(name_csv, 'Accuracy', index, [str(acc_experiment)])
        save_results(name_csv, 'Sensitivity', index, [str(sen_experiment)])
        save_results(name_csv, 'Specificity', index, [str(spe_experiment)])
        save_results(name_csv, 'AUC', index, [str(auc_experiment)])
        save_results(name_csv, 'F1Score', index, [str(f1_experiment)])
        
        if not kernel_linear:
            G_experiment = stats.mode(best_G_array)[0][0]
            best_G_experiments.append(G_experiment)
            save_results(name_csv, 'C and Gamma', index, [str(C_experiment), str(G_experiment)])
        else:
            save_results(name_csv, 'C', index, [str(C_experiment)])
            
    
    save_results(name_csv, ' ', ' ', [' '])        
    save_results(name_csv, ' ', ' ', ['Mean', 'Std'])
    save_results(name_csv, 'Accuracy', 'Final', [str(np.mean(accuracy_experiments)), str(np.std(accuracy_experiments))])
    save_results(name_csv, 'Sensitivity', 'Final', [str(np.mean(sensibility_experiments)), str(np.std(sensibility_experiments))])
    save_results(name_csv, 'Specificity', 'Final', [str(np.mean(specificity_experiments)), str(np.std(specificity_experiments))])
    save_results(name_csv, 'F1Score', 'Final', [str(np.mean(f1score_experiments)), str(np.std(f1score_experiments))])    
    save_results(name_csv, 'AUC', 'Final', [str(np.mean(auc_experiments)), str(np.std(auc_experiments))])    
    save_results(name_csv, 'C', 'Final', [str(np.mean(best_C_experiments)), str(np.std(best_C_experiments))])
    save_results(name_csv, ' ', ' ', [' '])
    if kernel_linear:
        save_results(name_csv, ' ', ' ', ['C'])
        save_results(name_csv, 'Parameters Mode', 'Final', [str(stats.mode(best_C_experiments)[0][0])])
        save_results(name_csv, 'Samples', 'Final', [samples])
    else:        
        save_results(name_csv, 'G', 'Final', [str(np.mean(best_G_experiments)), str(np.std(best_G_experiments))])
        save_results(name_csv, ' ', ' ', ['C', 'Gamma'])
        save_results(name_csv, 'Parameters Mode', 'Final', [str(stats.mode(best_C_experiments)[0][0]), str(stats.mode(best_G_experiments)[0][0])])
        save_results(name_csv, 'Samples', 'Final', [samples])

#%%
def Cross_validation_SVM_developSet(X,y,folds,name_csv,iterations=10,number_folds_validation=5,
                         max_c=100,min_c=0.001,number_c=6,max_gamma=10000,
                         min_gamma=0.00001,number_gamma=10,kernel_linear=False,
                         n_job = -1, shuffle_bool=True, balance=True):
    
    if os.path.exists(name_csv):
        os.remove(name_csv)
    
    
    C_array =np.geomspace(min_c,max_c,number_c)
    G_array=np.geomspace(min_gamma,max_gamma,number_gamma)
    
    if balance:
        X, y = balanceDataBase(X, y)
                
    samples = len(y)
    
    
    accuracy_experiments = []
    sensibility_experiments = []
    specificity_experiments = []
    f1score_experiments = []
    best_C_experiments = []
    best_G_experiments = []
    auc_experiments = []
    
    accuracy_test_experiments = []
    sensibility_test_experiments = []
    specificity_test_experiments = []
    f1score_test_experiments = []
    auc_test_experiments = []
    
    
    for index in range(iterations):
        index = index + 1
        score_array = []
        y_pred_array = []
        y_real_array = []
        best_C_array = []
        best_G_array = []
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, shuffle=True)
        print('# iteraccion: '+str(index))
        with tqdm(total = 100, file = sys.stdout) as pbar:
            step = 100/folds
            kf = model_selection.KFold(n_splits=folds,shuffle=shuffle_bool)
            for train_index, test_index in kf.split(X_train):
                
                X_train_inter, X_develop = X_train[train_index], X_train[test_index]
                y_train_intern, y_develop = y_train[train_index], y_train[test_index]

                
                normalize_transfor=preprocessing.StandardScaler().fit(X_train_inter)
                X_train_intern_N,X_develop_N=normalize_transfor.transform(X_train_inter),normalize_transfor.transform(X_develop)
                
                if kernel_linear:
                    parameters = {'kernel':['linear','rbf'], 'C':C_array}
                    svc = svm.SVC(gamma = 'scale')
                    clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
                    clf.fit(X_train_intern_N,y_train_intern)                
                    best_C=clf.best_estimator_.C
                    best_G='scale'
                
                else:
                    parameters = {'kernel':['rbf'], 'C':C_array,'gamma':G_array }
                    svc = svm.SVC()
                    clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
                    clf.fit(X_train_intern_N,y_train_intern)                
                    best_C=clf.best_estimator_.C
                    best_G=clf.best_estimator_.gamma
                
                #Best parameters and testSet
                
                SVM_Classifier = svm.SVC(kernel = clf.best_estimator_.kernel , gamma=best_G, C=best_C)
                SVM_Classifier.fit(X_train_intern_N,y_train_intern)
                y_pred=SVM_Classifier.predict(X_develop_N)
                score_test = SVM_Classifier.decision_function(X_develop_N)
                
                # Results in each fold
                
                y_real_array.append(y_develop)
                y_pred_array.append(y_pred)
                score_array.append(score_test)
                best_C_array.append(best_C)
                best_G_array.append(best_G)
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
        C_experiment = stats.mode(best_C_array)[0][0]
        auc_experiment = metrics.auc(fpr,tpr)



        
        
        accuracy_experiments.append(acc_experiment)
        sensibility_experiments.append(sen_experiment)
        specificity_experiments.append(spe_experiment)
        f1score_experiments.append(f1_experiment)
        auc_experiments.append(auc_experiment)
        best_C_experiments.append(C_experiment)

        
        save_results(name_csv, 'y_develop', index, list(y_real_total))
        save_results(name_csv, 'y_pred', index, list(y_pred_total))
        save_results(name_csv, 'score', index, list(score_pred_total))
        save_results(name_csv, 'Accuracy_develop', index, [str(acc_experiment)])
        save_results(name_csv, 'Sensitivity_develop', index, [str(sen_experiment)])
        save_results(name_csv, 'Specificity_develop', index, [str(spe_experiment)])
        save_results(name_csv, 'AUC_develop', index, [str(auc_experiment)])
        save_results(name_csv, 'F1Score_develop', index, [str(f1_experiment)])
        
        if not kernel_linear:
            G_experiment = stats.mode(best_G_array)[0][0]
            best_G_experiments.append(G_experiment)
            save_results(name_csv, 'C and Gamma', index, [str(C_experiment), str(G_experiment)])
            results_test = SimpleClassifier(X_train, y_train, X_test, y_test, {'Gamma': G_experiment, 'C': C_experiment}, 'rbf')

        else:
            save_results(name_csv, 'C', index, [str(C_experiment)])
            results_test = SimpleClassifier(X_train, y_train, X_test, y_test, {'C': C_experiment}, 'linear')

        save_results(name_csv, 'Accuracy_test', index, [str(results_test['acc'])])
        save_results(name_csv, 'Sensitivity_test', index, [str(results_test['sen'])])
        save_results(name_csv, 'Specificity_test', index, [str(results_test['spec'])])
        save_results(name_csv, 'AUC_test', index, [str(results_test['auc'])])
        save_results(name_csv, 'F1Score_test', index, [str(results_test['F-score'])])
        
        accuracy_test_experiments.append(results_test['acc'])
        sensibility_test_experiments.append(results_test['sen'])
        specificity_test_experiments.append(results_test['spec'])
        f1score_test_experiments.append(results_test['F-score'])
        auc_test_experiments.append(results_test['auc'])


    
    save_results(name_csv, ' ', ' ', [' '])        
    save_results(name_csv, ' ', ' ', ['Mean', 'Std'])
    save_results(name_csv, 'Accuracy_dev', 'Final', [str(np.mean(accuracy_experiments)), str(np.std(accuracy_experiments))])
    save_results(name_csv, 'Sensitivity_dev', 'Final', [str(np.mean(sensibility_experiments)), str(np.std(sensibility_experiments))])
    save_results(name_csv, 'Specificity_dev', 'Final', [str(np.mean(specificity_experiments)), str(np.std(specificity_experiments))])
    save_results(name_csv, 'F1Score_dev', 'Final', [str(np.mean(f1score_experiments)), str(np.std(f1score_experiments))])    
    save_results(name_csv, 'AUC_dev', 'Final', [str(np.mean(auc_experiments)), str(np.std(auc_experiments))])    
    save_results(name_csv, 'C', 'Final', [str(np.mean(best_C_experiments)), str(np.std(best_C_experiments))])
    save_results(name_csv, ' ', ' ', [' '])
    if kernel_linear:
        save_results(name_csv, ' ', ' ', ['C'])
        save_results(name_csv, 'Parameters Mode', 'Final', [str(stats.mode(best_C_experiments)[0][0])])
        save_results(name_csv, 'Samples', 'Final', [samples])
    else:        
        save_results(name_csv, 'G', 'Final', [str(np.mean(best_G_experiments)), str(np.std(best_G_experiments))])
        save_results(name_csv, ' ', ' ', ['C', 'Gamma'])
        save_results(name_csv, 'Parameters Mode', 'Final', [str(stats.mode(best_C_experiments)[0][0]), str(stats.mode(best_G_experiments)[0][0])])
        save_results(name_csv, 'Samples', 'Final', [samples])

    save_results(name_csv, ' ', ' ', [' '])        
    save_results(name_csv, ' ', ' ', ['Mean', 'Std'])
    save_results(name_csv, 'Accuracy_test', 'Final', [str(np.mean(accuracy_test_experiments)), str(np.std(accuracy_test_experiments))])
    save_results(name_csv, 'Sensitivity_test', 'Final', [str(np.mean(sensibility_test_experiments)), str(np.std(sensibility_test_experiments))])
    save_results(name_csv, 'Specificity_test', 'Final', [str(np.mean(specificity_test_experiments)), str(np.std(specificity_test_experiments))])
    save_results(name_csv, 'F1Score_test', 'Final', [str(np.mean(f1score_test_experiments)), str(np.std(f1score_test_experiments))])    
    save_results(name_csv, 'AUC_test', 'Final', [str(np.mean(auc_test_experiments)), str(np.std(auc_test_experiments))])    
    save_results(name_csv, ' ', ' ', [' '])


#%%
def eerMeasures(experiments):

    eerMatriz = {}
    eerZ = np.zeros((len(experiments[0][:,0]), len(experiments)), dtype=bool)
    mc_eer = {}
    measure_eer = {}
     
    for index, experim in enumerate(experiments):
    #itera sobre los 10 experimentos
            #fpr, tpr, thr = roc_curve(experiments[4][:,1], experiments[4][:,0], pos_label=1)
        fpr, tpr, thr = metrics.roc_curve(experim[:,1], experim[:,0], pos_label=1)
        fnr = 1 - tpr
        eer_threshold = thr[np.nanargmin(np.absolute((fnr - fpr)))] #Threshold hace mínima las distancia entre fnr y fpr en valor absoluto
                    # The EER is defined as FPR = 1 - PTR = FNR
        EER1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
                    # Check the value should be close to the previous
        EER2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        eerMatriz[index] = [fpr,tpr,thr,fnr, eer_threshold,EER1,EER2]
       
        eerZ[:,index] = experim[:,0] > eer_threshold
       
        mc_eer[index] = metrics.confusion_matrix(experim[:,1],eerZ[:,index])
   
        acc_eer = metrics.accuracy_score(experim[:,1],eerZ[:,index])
#        acc_balan = balanced_accuracy_score(experim[:,1],eerZ[:,index])
        param_eer = metrics.precision_recall_fscore_support(experim[:,1], eerZ[:,index], pos_label=1, average='weighted')
        sensi_eer = (mc_eer[index][1][1])/(mc_eer[index][1][1]+mc_eer[index][1][0])
        speci_eer = (mc_eer[index][0][0])/(mc_eer[index][0][0]+mc_eer[index][0][1])
        f1_eer = param_eer[2]
        auc_eer = metrics.auc(fpr, tpr)
        
       
        measure_eer[index+1] = [[1 if eer else 0 for eer in eerZ[:,index]], acc_eer, sensi_eer, speci_eer, f1_eer, auc_eer, eer_threshold]
    return measure_eer


def Cross_validation_SVM_eer(X,y,folds,name_csv,iterations=10,number_folds_validation=5,
                         max_c=100,min_c=0.001,number_c=6,max_gamma=1000,
                         min_gamma=0.0001,number_gamma=8,kernel_linear=False,
                         n_job = -1, shuffle_bool=True):
    
    if os.path.exists(name_csv):
        os.remove(name_csv)
    
    
    C_array =np.geomspace(min_c,max_c,number_c)
    G_array=np.geomspace(min_gamma,max_gamma,number_gamma)
    
    

    experiments = []
    y_experiments = {}
    
    
    for index in range(iterations):
        
        index = index + 1
        y_experiments[index] = {}
        score_array = []
        y_pred_array = []
        y_real_array = []
        best_C_array = []
        best_G_array = []
  
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
                
                if kernel_linear:
                    parameters = {'kernel':['linear','rbf'], 'C':C_array}
                else:
                    parameters = {'kernel':['rbf'], 'C':C_array,'gamma':G_array }
            
  
                svc = svm.SVC()
                clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
                clf.fit(X_train,y_train)                
                best_C=clf.best_estimator_.C
                best_G=clf.best_estimator_.gamma
                
                #Best parameters and testSet
                
                SVM_Classifier = svm.SVC(kernel = clf.best_estimator_.kernel , gamma=best_G, C=best_C)
                SVM_Classifier.fit(X_train_N,y_train)
                y_pred=SVM_Classifier.predict(X_test_N)
                score_test = SVM_Classifier.decision_function(X_test_N)
                
                # Results in each fold
                
                y_real_array.append(y_test)
                y_pred_array.append(y_pred)
                score_array.append(score_test)
                best_C_array.append(best_C)
                best_G_array.append(best_G)
                pbar.update(step)
            
        y_experiments[index]['score'] = np.hstack(score_array)
        y_experiments[index]['y_real']= np.hstack(y_real_array)
        y_experiments[index]['C'] = stats.mode(best_C_array)[0][0]
        y_experiments[index]['G'] = stats.mode(best_G_array)[0][0]
        
        
        experim = np.column_stack([np.hstack(score_array), np.hstack(y_real_array), np.hstack(y_pred_array)])
        
        experiments.append(experim)
        
    dic_results = eerMeasures(experiments)
    
    for key, value in dic_results.items():
        save_results(name_csv, 'y_real', index, list(y_experiments[key]['y_real']))
        save_results(name_csv, 'y_pred', index, list(value[0]))
        save_results(name_csv, 'score', index, list(y_experiments[key]['score']))
        save_results(name_csv, 'Accuracy', index, [str(value[1])])
        save_results(name_csv, 'Sensitivity', index, [str(value[2])])
        save_results(name_csv, 'Specificity', index, [str(value[3])])
        save_results(name_csv, 'AUC', index, [str(value[5])])
        save_results(name_csv, 'F1Score', index, [str(value[4])])
        save_results(name_csv, 'Threshold', index, [str(value[6])])
        save_results(name_csv, 'C and Gamma', index, [str(y_experiments[key]['C']), str(y_experiments[key]['G'])])
        
    save_results(name_csv, ' ', ' ', [' '])        
    save_results(name_csv, ' ', ' ', ['Mean', 'Std'])
    save_results(name_csv, 'Accuracy', 'Final', [str(np.mean([value[1] for value in dic_results.values()])), str(np.std([value[1] for value in dic_results.values()]))])
    save_results(name_csv, 'Sensitivity', 'Final', [str(np.mean([value[2] for value in dic_results.values()])), np.std([value[2] for value in dic_results.values()])])
    save_results(name_csv, 'Specificity', 'Final', [np.mean([value[3] for value in dic_results.values()]), np.std([value[3] for value in dic_results.values()])])
    save_results(name_csv, 'F1Score', 'Final', [np.mean([value[4] for value in dic_results.values()]), np.std([value[4] for value in dic_results.values()])])    
    save_results(name_csv, 'AUC', 'Final', [np.mean([value[5] for value in dic_results.values()]), np.std([value[5] for value in dic_results.values()])])    
    save_results(name_csv, 'C', 'Final', [np.mean([value['C'] for value in y_experiments.values()]), np.std([value['C'] for value in y_experiments.values()])])
    save_results(name_csv, 'G', 'Final', [np.mean([value['G'] for value in y_experiments.values()]), np.std([value['G'] for value in y_experiments.values()])])
    save_results(name_csv, 'Threshold', 'Final', [np.mean([value[6] for value in dic_results.values()]), np.std([value[6] for value in dic_results.values()])])
    save_results(name_csv, ' ', ' ', [' '])
    save_results(name_csv, ' ', ' ', ['C', 'Gamma'])
    save_results(name_csv, 'Parameters Mode', 'Final', [stats.mode([value['C'] for value in y_experiments.values()])[0][0], stats.mode([value['G'] for value in y_experiments.values()])[0][0]])


def Cross_validation_SVMMulticlass(X,y,folds,name_csv,iterations=10,number_folds_validation=10,
                         max_c=100,min_c=0.001,number_c=6,max_gamma=10000,
                         min_gamma=0.00001,number_gamma=10,kernel_linear=False,
                         n_job = -1, shuffle_bool=True, balance=True):
    
    if os.path.exists(name_csv):
        os.remove(name_csv)
    
    
    C_array =np.geomspace(min_c,max_c,number_c)
    G_array=np.geomspace(min_gamma,max_gamma,number_gamma)
    
    if balance:
        X, y = balanceDataBaseMulticlass(X, y)
                
    samples = len(y)
    
    
    accuracy_experiments = []
    f1score_experiments = []
    kappa_score_experiments = []
    best_C_experiments = []
    best_G_experiments = []
    
    for index in range(iterations):
        index = index + 1
        y_pred_array = []
        y_real_array = []
        best_C_array = []
        best_G_array = []
        

        print('# iteraccion: '+str(index))
        with tqdm(total = 100, file = sys.stdout) as pbar:
            step = 100/folds
            kf = model_selection.KFold(n_splits=folds,shuffle=shuffle_bool)
            for train_index, test_index in kf.split(X):
                
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                
                normalize_transfor=preprocessing.StandardScaler().fit(X_train)
                X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
                
                if kernel_linear:
                    parameters = {'kernel':['linear','rbf'], 'C':C_array}
                    svc = svm.SVC(gamma = 'scale')
                    clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
                    clf.fit(X_train,y_train)                
                    best_C=clf.best_estimator_.C
                    best_G='scale'
                
                else:
                    parameters = {'kernel':['rbf'], 'C':C_array,'gamma':G_array }
                    svc = svm.SVC()
                    clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
                    clf.fit(X_train,y_train)                
                    best_C=clf.best_estimator_.C
                    best_G=clf.best_estimator_.gamma
                
                #Best parameters and testSet
                
                SVM_Classifier = svm.SVC(kernel = clf.best_estimator_.kernel , gamma=best_G, C=best_C)
                SVM_Classifier.fit(X_train_N,y_train)
                y_pred=SVM_Classifier.predict(X_test_N)
                score_test = SVM_Classifier.decision_function(X_test_N)
                
                # Results in each fold
                
                y_real_array.append(y_test)
                y_pred_array.append(y_pred)
#                score_array.append(score_test)
                best_C_array.append(best_C)
                best_G_array.append(best_G)
                pbar.update(step)
            
#        score_pred_total = np.hstack(score_array)
        y_real_total = np.hstack(y_real_array)
        y_pred_total = np.hstack(y_pred_array)

        
        acc_experiment = metrics.accuracy_score(y_real_total, y_pred_total)*100

        f1_experiment = metrics.f1_score(y_real_total,y_pred_total, average = 'weighted')*100
        kapa_score = metrics.cohen_kappa_score(y_real_total, y_pred_total)
        C_experiment = stats.mode(best_C_array)[0][0]

        
        accuracy_experiments.append(acc_experiment)

        f1score_experiments.append(f1_experiment)
        kappa_score_experiments.append(kapa_score)
        best_C_experiments.append(C_experiment)

        
        save_results(name_csv, 'y_real', index, list(y_real_total))
        save_results(name_csv, 'y_pred', index, list(y_pred_total))
#        save_results(name_csv, 'score', index, list(score_pred_total))
        save_results(name_csv, 'Accuracy', index, [acc_experiment])
        save_results(name_csv, 'F1Score', index, [f1_experiment])
        save_results(name_csv, 'KappaScore', index, [kapa_score])
        
        if not kernel_linear:
            G_experiment = stats.mode(best_G_array)[0][0]
            best_G_experiments.append(G_experiment)
            save_results(name_csv, 'C and Gamma', index, [C_experiment, G_experiment])
        else:
            save_results(name_csv, 'C', index, [C_experiment])
            
    
    save_results(name_csv, ' ', ' ', [' '])        
    save_results(name_csv, ' ', ' ', ['Mean', 'Std'])
    save_results(name_csv, 'Accuracy', 'Final', [np.mean(accuracy_experiments), np.std(accuracy_experiments)])
    save_results(name_csv, 'F1Score', 'Final', [np.mean(f1score_experiments), np.std(f1score_experiments)])    
    save_results(name_csv, 'KapaScore', 'Final', [np.mean(kappa_score_experiments), np.std(kappa_score_experiments)])    
    save_results(name_csv, 'C', 'Final', [np.mean(best_C_experiments), np.std(best_C_experiments)])
    save_results(name_csv, ' ', ' ', [' '])
    if kernel_linear:
        save_results(name_csv, 'G', 'Final', [np.mean(best_G_experiments), np.std(best_G_experiments)])
        save_results(name_csv, ' ', ' ', ['C'])
        save_results(name_csv, 'Parameters Mode', 'Final', [stats.mode(best_C_experiments)[0][0]])
        save_results(name_csv, 'Samples', 'Final', [samples])
    else:        
        save_results(name_csv, ' ', ' ', ['C', 'Gamma'])
        save_results(name_csv, 'Parameters Mode', 'Final', [stats.mode(best_C_experiments)[0][0], stats.mode(best_G_experiments)[0][0]])
        save_results(name_csv, 'Samples', 'Final', [samples])

# Feat_GloVe = np.loadtxt('/home/daniel/Escritorio/GITA/GeneralCode/Features-GloVe-300.txt')[0:20, 0:5]
# #Load labels
# path_base='LabelsClassification.csv'
# labels_data = pd.read_csv(path_base,delimiter=',')
# y = list(labels_data['Agr'])[0:20] 
# y = np.asarray(y)

def SimpleClassifier(X_train, Y_train, X_test, Y_test, params, kernel):
    normalize_transfor=preprocessing.StandardScaler().fit(X_train)
    X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
    
    results = {}
    #Best parameters and testSet
    if kernel=='linear':
        SVM_Classifier = svm.SVC(kernel = kernel, C=params['C'])
        SVM_Classifier.fit(X_train_N,Y_train)
        y_pred=SVM_Classifier.predict(X_test_N)
        score_test = SVM_Classifier.decision_function(X_test_N)
    else:
        SVM_Classifier = svm.SVC(kernel = kernel, gamma=params['Gamma'], C=params['C'])
        SVM_Classifier.fit(X_train_N,Y_train)
        y_pred=SVM_Classifier.predict(X_test_N)
        score_test = SVM_Classifier.decision_function(X_test_N)

        tn, fp, fn, tp = metrics.confusion_matrix(Y_test, y_pred).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, score_test)
        
        results['acc'] = metrics.accuracy_score(Y_test, y_pred)*100
        results['sen'] = tp/(tp+fn)*100
        results['spec'] = tn/(tn+fp)*100
        results['F-score'] = metrics.f1_score(Y_test,y_pred)*100
        results['auc'] = metrics.auc(fpr,tpr)
    
    return results
    # Results in each fold
    
