#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:14:45 2018

@author: Cristian Rios
"""
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score



def XGBoost_kfold (matriz, labels, n_repet, n_kfold, shuffle=True):
    
    #Parametros y variables a usar 
    
    learning_rate = [0.001, 0.01, 0.1, 0.3]
    n_estimators = [50, 100, 200, 300]
    max_depth = [1, 3, 5, 7]
    
    lr_total=[]
    estimators_total=[]
    max_depth_total=[]
    acc_total=[]
    acc_bal_total=[]
    sensi_total=[]
    speci_total=[]
    f1_total=[]
    auc_total=[]
    
    y_scores_final=[]
    z_final=[]
    y_final=[]

    for ite in range (0,n_repet):
        print("------------------- interaction = "+str(ite)+" --------------------")
      
        y_scores=[]
        z_total=[]
        y_total=[]
    
        kf = KFold(n_splits=n_kfold,shuffle=shuffle)
        kf.get_n_splits(matriz)
    
        for train, test in kf.split(matriz):
            X_train, X_test = matriz[train], matriz[test]
            y_train, y_test = labels[train], labels[test]
    
            #Restar la media y dividir la std
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
    
            #Calculo de hiperparametros
            parameters = {'learning_rate':learning_rate, 'n_estimators':n_estimators, 'max_depth': max_depth}
            xgb_model = xgb.XGBClassifier(objective="binary:logistic")
            clf = GridSearchCV(xgb_model, parameters,n_jobs=-1,cv=4)
            clf.fit(X_train, y_train)
            lr_ideal=clf.best_estimator_.learning_rate
            estimators_ideal=clf.best_estimator_.n_estimators
            max_depth_ideal=clf.best_estimator_.max_depth
            lr_total.append(lr_ideal)
            estimators_total.append(estimators_ideal)
            max_depth_total.append(max_depth_ideal)
    
            #Entrenamiento y prediccion
            xgb_model = xgb.XGBClassifier(objective="binary:logistic", learning_rate=lr_ideal,
                                          n_estimators=estimators_ideal, max_depth=max_depth_ideal)

            xgb_model.fit(X_train, y_train)
            X_test=scaler.transform(X_test)
            Z=xgb_model.predict(X_test)
                        
            # Guardar score, predic y etiqueta
            y_scores.append(xgb_model.predict_proba(X_test)[:,1])
            z_total.append(Z)
            y_total.append(y_test)
            
        y_total=np.hstack(y_total)
        z_total=np.hstack(z_total)
        y_scores=np.hstack(y_scores)
        
        #Calculo de metricas
        acc=accuracy_score(y_total,z_total)
        acc_balan=balanced_accuracy_score(y_total,z_total)
        c_m=confusion_matrix(y_total,z_total)
        param=precision_recall_fscore_support(y_total, z_total, pos_label=1, average='weighted')
        sensi=(c_m[1][1])/(c_m[1][1]+c_m[1][0])
        speci=(c_m[0][0])/(c_m[0][0]+c_m[0][1])
        f1=param[2]
        fpr, tpr, thresholds = metrics.roc_curve(y_total, y_scores)
        auc=metrics.auc(fpr, tpr)
        
        print("Accuracy=",acc*100)
        print("Accuracy balanced=",acc_balan*100)
        print("Sensivity = ", sensi)
        print("Specificity = ", speci)
        print("F1score = ", f1)
        print("AUC= ", auc)
    
        y_final.append(y_total)
        z_final.append(z_total)
        y_scores_final.append(y_scores)

        acc_total.append(acc*100)
        acc_bal_total.append(acc_balan*100)
        sensi_total.append(sensi)
        speci_total.append(speci)
        f1_total.append(f1)
        auc_total.append(auc)
    
    print("------------------------ FINAL -------------------------")
    print("Total learning_rate = ", np.str(stats.mode(lr_total)[0][0]))
    print("Total n_estimators = ", np.str(stats.mode(estimators_total)[0][0]))
    print("Total max_depth = ", np.str(stats.mode(max_depth_total)[0][0]))
    print("Total Accuracy = " + np.str(np.mean(acc_total)) + " +/- " + np.str(np.std(acc_total)))
    print("Total Accuracy Balanced = " + np.str(np.mean(acc_bal_total)) + " +/- " + np.str(np.std(acc_bal_total)))
    print("Total Sensitivity = " + np.str(np.mean(sensi_total)) + " +/- " + np.str(np.std(sensi_total)))
    print("Total Recall = " + np.str(np.mean(speci_total)) + " +/- " + np.str(np.std(speci_total)))
    print("Total F1score = " + np.str(np.mean(f1_total)) + " +/- " + np.str(np.std(f1_total)))
    print("Total AUC = " + np.str(np.mean(auc_total)) + " +/- " + np.str(np.std(auc_total)))

    return lr_total, estimators_total, max_depth_total, acc_total, acc_bal_total, sensi_total, speci_total, f1_total, auc_total, y_final, z_final, y_scores_final
