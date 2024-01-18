#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 09:38:42 2023

@author: daniel
"""

from sklearn import preprocessing, svm, model_selection, metrics
import numpy as np
from scipy import stats

class ResultsGeneral_biclass_regression_cv():
    def __init__(self):
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, best_C, best_G, normalizer, model):
        dictResults = {}
        
        dictResults['MSE'] = metrics.mean_squared_error(y_real_total, y_pred_total)
        
        dictResults['MAE'] = metrics.mean_absolute_error(y_real_total, y_pred_total)
        dictResults['Pearson'] = stats.pearsonr(y_real_total, y_pred_total)
        dictResults['Spearman'] = stats.spearmanr(y_real_total, y_pred_total)
        dictResults['C'] = best_C
        dictResults['Gamma'] = best_G

        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['model'] = model


        self.ResultsByFold[keyFold] = dictResults
        
    def computeGeneralResults(self):
        dict_folds = list(self.ResultsByFold.values())
        results_folds = {}
        for key in dict_folds[0].keys():
            results_folds[key] = [d_aux[key] for d_aux in dict_folds]
            
        for key_target in list(dict_folds[0].keys())[0:5]:
            
            self.ResultsGeneral[key_target+' mean'] = np.mean(results_folds[key_target]) 
            self.ResultsGeneral[key_target+' std'] = np.std(results_folds[key_target]) 
        
        self.ResultsGeneral['y_real_complete'] = np.hstack(results_folds['y_real'])
        self.ResultsGeneral['y_pred_complete'] = np.hstack(results_folds['y_pred'])
        self.ResultsGeneral['best C'] = stats.mode(results_folds['C'])[0][0]
        self.ResultsGeneral['best Gamma'] = stats.mode(results_folds['Gamma'])[0][0]        
        self.ResultsGeneralByFold = results_folds


def SimpleRegressor(X_train, Y_train, X_test, Y_test, number_folds_validation, results_object, keyFold, kernel_linear=False, n_job=-1):
    normalize_transfor=preprocessing.StandardScaler().fit(X_train)
    X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
    
    C_array =np.geomspace(1e-4,1e3,8)
    G_array=np.geomspace(1e-6,1e6,13)
    
    #Best parameters and testSet
    if kernel_linear:
        parameters = {'kernel':['linear'], 'C':C_array}
        svc = svm.SVR(class_weight = 'balanced')
        clf_regresor = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job, scoring='r2')
        clf_regresor.fit(X_train_N,Y_train)                
        best_C=clf_regresor.best_estimator_.C
        best_G='scale'
    
    else:
        parameters = {'kernel':['rbf'], 'C':C_array,'gamma':G_array }
        svc = svm.SVR()
        clf_regressor = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job, scoring='r2')
        clf_regressor.fit(X_train_N,Y_train)                
        best_C=clf_regressor.best_estimator_.C
        best_G=clf_regressor.best_estimator_.gamma
    
    print('####'+str(clf_regressor.best_estimator_.kernel)+'#####')
    if kernel_linear:
        SVR_regression = svm.SVR(kernel = clf_regressor.best_estimator_.kernel , C=best_C)
    else:    
        SVR_regression = svm.SVR(kernel = clf_regressor.best_estimator_.kernel , gamma=best_G, C=best_C)
    
    SVR_regression.fit(X_train_N,Y_train)

    y_pred=SVR_regression.predict(X_test_N)
 #   score_test = SVM_Classifier.decision_function(X_test_N) 
    
    results_object.AddResultsInFold(keyFold, Y_test, y_pred, best_C, best_G, normalize_transfor, SVR_regression)
    

    return results_object