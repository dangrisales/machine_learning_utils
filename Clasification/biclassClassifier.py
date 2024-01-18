#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:56:17 2022

@author: daniel
"""

from sklearn import preprocessing, svm, model_selection,metrics, linear_model
import numpy as np
from scipy import stats
import xgboost as xgb

class ResultsGeneral_biclass_experiments_cv():
    def __init__(self, iterations_number=0):
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsIteration = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        self.iterations_n = iterations_number
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, ids_test, params_dict, score_pred_total, normalizer, model):
        dictResults = {}
        tn, fp, fn, tp = metrics.confusion_matrix(y_real_total, y_pred_total).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_real_total,score_pred_total)
        
        
        dictResults['Acc'] = metrics.accuracy_score(y_real_total, y_pred_total)*100
        
        dictResults['Sen'] = tp/(tp+fn)*100
        dictResults['Spe'] = tn/(tn+fp)*100
        dictResults['F1'] = metrics.f1_score(y_real_total,y_pred_total)*100
        dictResults['auc'] = metrics.auc(fpr,tpr)
        dictResults['parms'] = params_dict

        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['score'] = score_pred_total
        dictResults['ids_test'] = ids_test
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
        self.ResultsGeneral['y_score_complete'] = np.hstack(results_folds['score'])        
        self.ResultsGeneralByFold = results_folds


    def computeGeneralResultsByIteration(self):
        dict_folds = self.ResultsByFold
                
        results_folds = {}
        for key in dict_folds[0].keys():
            results_folds[key] = [d_aux[key] for d_aux in dict_folds]
            
        for key_target in list(dict_folds[0].keys())[0:5]:
            self.ResultsGeneral = {}
            self.ResultsGeneral[key_target+' mean'] = np.mean(results_folds[key_target]) 
            self.ResultsGeneral[key_target+' std'] = np.std(results_folds[key_target]) 
        
        self.ResultsGeneral['y_real_complete'] = np.hstack(results_folds['y_real'])
        self.ResultsGeneral['y_pred_complete'] = np.hstack(results_folds['y_pred'])
        self.ResultsGeneral['y_score_complete'] = np.hstack(results_folds['score'])        
        self.ResultsGeneralByFold = results_folds


class ClassifierResultsBiclassCrossValidation():
    def __init__(self, iter_n=0):
        self.ResultsByIter = {}
        self.ResultsGeneralIter = {}
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        self.iterations_n = iter_n
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, ids_test, best_C, best_G, score_pred_total, normalizer, model):
        dictResults = {}
        tn, fp, fn, tp = metrics.confusion_matrix(y_real_total, y_pred_total).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_real_total,score_pred_total)
        
        
        dictResults['Acc'] = metrics.accuracy_score(y_real_total, y_pred_total)*100
        
        dictResults['Sen'] = tp/(tp+fn)*100
        dictResults['Spe'] = tn/(tn+fp)*100
        dictResults['F1'] = metrics.f1_score(y_real_total,y_pred_total)*100
        dictResults['auc'] = metrics.auc(fpr,tpr)
        dictResults['C'] = best_C
        dictResults['Gamma'] = best_G

        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['score'] = score_pred_total
        dictResults['ids_in_fold'] = ids_test
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
        self.ResultsGeneral['y_score_complete'] = np.hstack(results_folds['score'])
        self.ResultsGeneral['best C'] = stats.mode(results_folds['C'])[0][0]
        self.ResultsGeneral['best Gamma'] = stats.mode(results_folds['Gamma'])[0][0]        
        self.ResultsGeneralByFold = results_folds

    def computeGeneralResultsByIterations(self):
        dict_folds = self.ResultsByFold
        iterations_number = self.iterations_n
        

        for i in range(iterations_number):
            dic_iter = {}
            results_by_iter = [[values['y_real'], values['y_pred'], np.asarray(values['score']), values['ids_in_fold']] for key, values in dict_folds.items() if 'ite_'+str(i) in key]
            
            y_real_ite = np.hstack([results_by_iter[i][0] for i in range(len(results_by_iter))]) #results_by_iter[0,:]
            y_pred_ite =  np.hstack([results_by_iter[i][1] for i in range(len(results_by_iter))])
            y_score_ite = np.hstack([results_by_iter[i][2] for i in range(len(results_by_iter))])
            ids_in_fold = np.hstack([results_by_iter[i][3] for i in range(len(results_by_iter))])

            tn, fp, fn, tp = metrics.confusion_matrix(y_real_ite, y_pred_ite).ravel()
            fpr, tpr, thresholds = metrics.roc_curve(y_real_ite,y_score_ite)        
            dic_iter['Acc'] = metrics.accuracy_score(y_real_ite, y_pred_ite)*100
            
            dic_iter['Sen'] = tp/(tp+fn)*100
            dic_iter['Spe'] = tn/(tn+fp)*100
            dic_iter['F1'] = metrics.f1_score(y_real_ite,y_pred_ite)*100
            dic_iter['auc'] = metrics.auc(fpr,tpr)
    
            dic_iter['y_real'] = y_real_ite
            dic_iter['y_pred'] = y_pred_ite
            dic_iter['score'] = y_score_ite
            dic_iter['ids_fold'] = ids_in_fold
            
            self.ResultsByIter['ite_'+str(i)] = dic_iter
        
        dict_iter = list(self.ResultsByIter.values())
        results_iter = {}
        for key in dict_iter[0].keys():
            results_iter[key] = [d_aux[key] for d_aux in dict_iter]
        
        key_compute = ['Acc', 'Sen', 'Spe', 'F1', 'auc']
            
        for key_target in key_compute: #Acc, Sen, Spe, F1, AUC
            
            self.ResultsGeneralIter[key_target+' mean'] = np.mean(results_iter[key_target]) 
            self.ResultsGeneralIter[key_target+' std'] = np.std(results_iter[key_target]) 


class ClassifierResultsBiclassCrossValidationGeneralModel():
    def __init__(self, iter_n=0):
        self.ResultsByIter = {}
        self.ResultsGeneralIter = {}
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        self.iterations_n = iter_n
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, ids_test, score_pred_total, ids_train, scores_train, best_parameters_dict, normalizer, model):
        dictResults = {}
        tn, fp, fn, tp = metrics.confusion_matrix(y_real_total, y_pred_total).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y_real_total,score_pred_total)
        
        
        dictResults['Acc'] = metrics.accuracy_score(y_real_total, y_pred_total)*100
        
        dictResults['Sen'] = tp/(tp+fn)*100
        dictResults['Spe'] = tn/(tn+fp)*100
        dictResults['F1'] = metrics.f1_score(y_real_total,y_pred_total)*100
        dictResults['auc'] = metrics.auc(fpr,tpr)
        dictResults['parameters'] = best_parameters_dict


        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['score'] = score_pred_total
        dictResults['ids_in_fold'] = ids_test
        dictResults['ids_in_fold_train'] = ids_train
        dictResults['scores_train'] = scores_train
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
        self.ResultsGeneral['y_score_complete'] = np.hstack(results_folds['score'])
       
        self.ResultsGeneralByFold = results_folds

    def computeGeneralResultsByIterations(self):
        dict_folds = self.ResultsByFold
        iterations_number = self.iterations_n
        

        for i in range(1,iterations_number+1):
            dic_iter = {}
            results_by_iter = [[values['y_real'], values['y_pred'], np.asarray(values['score']), values['ids_in_fold']] for key, values in dict_folds.items() if 'ite_'+str(i) in key]
            
            y_real_ite = np.hstack([results_by_iter[i][0] for i in range(len(results_by_iter))]) #results_by_iter[0,:]
            y_pred_ite =  np.hstack([results_by_iter[i][1] for i in range(len(results_by_iter))])
            y_score_ite = np.hstack([results_by_iter[i][2] for i in range(len(results_by_iter))])
            ids_in_fold = np.hstack([results_by_iter[i][3] for i in range(len(results_by_iter))])

            tn, fp, fn, tp = metrics.confusion_matrix(y_real_ite, y_pred_ite).ravel()
            fpr, tpr, thresholds = metrics.roc_curve(y_real_ite,y_score_ite)        
            dic_iter['Acc'] = metrics.accuracy_score(y_real_ite, y_pred_ite)*100
            
            dic_iter['Sen'] = tp/(tp+fn)*100
            dic_iter['Spe'] = tn/(tn+fp)*100
            dic_iter['F1'] = metrics.f1_score(y_real_ite,y_pred_ite)*100
            dic_iter['auc'] = metrics.auc(fpr,tpr)
    
            dic_iter['y_real'] = y_real_ite
            dic_iter['y_pred'] = y_pred_ite
            dic_iter['score'] = y_score_ite
            dic_iter['ids_fold'] = ids_in_fold
            
            self.ResultsByIter['ite_'+str(i)] = dic_iter
        
        dict_iter = list(self.ResultsByIter.values())
        results_iter = {}
        for key in dict_iter[0].keys():
            results_iter[key] = [d_aux[key] for d_aux in dict_iter]
        
        key_compute = ['Acc', 'Sen', 'Spe', 'F1', 'auc']
            
        for key_target in key_compute: #Acc, Sen, Spe, F1, AUC
            
            self.ResultsGeneralIter[key_target+' mean'] = np.mean(results_iter[key_target]) 
            self.ResultsGeneralIter[key_target+' std'] = np.std(results_iter[key_target]) 
        

class ClassifierResultsMulticlassCrossValidation():
    def __init__(self, iter_n=0):
        self.ResultsByIter = {}
        self.ResultsGeneralIter = {}
        self.ResultsByFold = {}
        self.ResultsGeneral = {}
        self.ResultsGeneralByFold = {}
        self.Model_dict = {}
        self.iterations_n = iter_n
    
            
        
    def AddResultsInFold(self, keyFold, y_real_total, y_pred_total, ids_test, score_pred_total, ids_train, scores_train, best_parameters_dict, normalizer, model):
        dictResults = {}

        
        
        dictResults['Acc'] = metrics.accuracy_score(y_real_total, y_pred_total)*100
        dictResults['F1'] = metrics.f1_score(y_real_total,y_pred_total, average='weighted')*100
        dictResults['UAR'] = metrics.balanced_accuracy_score(y_real_total,y_pred_total) * 100
        dictResults['parameters'] = best_parameters_dict

        dictResults['Normalizer'] = normalizer
        dictResults['y_real'] = y_real_total
        dictResults['y_pred'] = y_pred_total
        dictResults['ids_in_fold'] = ids_test
        dictResults['score'] = score_pred_total
        dictResults['model'] = model


        self.ResultsByFold[keyFold] = dictResults
        
    def computeGeneralResults(self):
        dict_folds = list(self.ResultsByFold.values())
        results_folds = {}
        for key in dict_folds[0].keys():
            results_folds[key] = [d_aux[key] for d_aux in dict_folds]
            
        for key_target in list(dict_folds[0].keys())[0:3]:
            
            self.ResultsGeneral[key_target+' mean'] = np.mean(results_folds[key_target]) 
            self.ResultsGeneral[key_target+' std'] = np.std(results_folds[key_target]) 
        
        self.ResultsGeneral['y_real_complete'] = np.hstack(results_folds['y_real'])
        self.ResultsGeneral['y_pred_complete'] = np.hstack(results_folds['y_pred'])
        self.ResultsGeneral['y_score_complete'] = np.vstack(results_folds['score'])
        # self.ResultsGeneral['best C'] = stats.mode(results_folds['C'])[0][0]
        # self.ResultsGeneral['best Gamma'] = stats.mode(results_folds['Gamma'])[0][0]
        
        self.ResultsGeneralByFold = results_folds        

    def computeGeneralResultsByIterations(self):
        dict_folds = self.ResultsByFold
        iterations_number = self.iterations_n
        

        for i in range(1,iterations_number+1):
            dic_iter = {}
            results_by_iter = [[values['y_real'], values['y_pred'], np.asarray(values['score']), values['ids_in_fold']] for key, values in dict_folds.items() if 'ite_'+str(i) in key]
            
            y_real_ite = np.hstack([results_by_iter[i][0] for i in range(len(results_by_iter))]) #results_by_iter[0,:]
            y_pred_ite =  np.hstack([results_by_iter[i][1] for i in range(len(results_by_iter))])
            #y_score_ite = np.hstack([results_by_iter[i][2] for i in range(len(results_by_iter))])
            ids_in_fold = np.hstack([results_by_iter[i][3] for i in range(len(results_by_iter))])

            #tn, fp, fn, tp = metrics.confusion_matrix(y_real_ite, y_pred_ite).ravel()
            #fpr, tpr, thresholds = metrics.roc_curve(y_real_ite,y_score_ite)        
            dic_iter['Acc'] = metrics.accuracy_score(y_real_ite, y_pred_ite)*100
            dic_iter['F1'] = metrics.f1_score(y_real_ite, y_pred_ite, average='weighted')*100
            dic_iter['UAR'] = metrics.balanced_accuracy_score(y_real_ite, y_pred_ite) * 100
    
    
            dic_iter['y_real'] = y_real_ite
            dic_iter['y_pred'] = y_pred_ite
            #dic_iter['score'] = y_score_ite
            dic_iter['ids_fold'] = ids_in_fold
            
            self.ResultsByIter['ite_'+str(i)] = dic_iter
        
        dict_iter = list(self.ResultsByIter.values())
        results_iter = {}
        for key in dict_iter[0].keys():
            results_iter[key] = [d_aux[key] for d_aux in dict_iter]
        
        key_compute = ['Acc', 'F1', 'UAR']
            
        for key_target in key_compute: #Acc, Sen, Spe, F1, AUC
            
            self.ResultsGeneralIter[key_target+' mean'] = np.mean(results_iter[key_target]) 
            self.ResultsGeneralIter[key_target+' std'] = np.std(results_iter[key_target]) 
        


def SimpleClassifier(X_train, Y_train, ids_train, X_test, Y_test, ids_test, number_folds_validation, results_object, keyFold, kernel_linear=False, n_job=-1):
    normalize_transfor=preprocessing.StandardScaler().fit(X_train)
    X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
    
    C_array =[2**(-3),2**(-2),2**(-1),2**(0),2**(1),2**(2),2**(3)]#np.geomspace(1e-4,1e3,8)
    G_array=np.geomspace(1e-6,1e6,13)
    
    #Best parameters and testSet
    if kernel_linear:
        parameters = {'kernel':['linear'], 'C':C_array}
        svc = svm.SVC(class_weight = 'balanced')
        clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
        clf.fit(X_train_N,Y_train)                
        best_C=clf.best_estimator_.C
        best_G='scale'
    
    else:
        parameters = {'kernel':['rbf'], 'C':C_array,'gamma':G_array }
        svc = svm.SVC(class_weight = 'balanced')
        clf = model_selection.GridSearchCV(svc, parameters, cv=number_folds_validation, n_jobs = n_job)
        clf.fit(X_train_N,Y_train)                
        best_C=clf.best_estimator_.C
        best_G=clf.best_estimator_.gamma
    
    print('####'+str(clf.best_estimator_.kernel)+'#####')
    if kernel_linear:
        SVM_Classifier = svm.SVC(kernel = clf.best_estimator_.kernel , C=best_C, class_weight = 'balanced')
    else:    
        SVM_Classifier = svm.SVC(kernel = clf.best_estimator_.kernel , gamma=best_G, C=best_C, class_weight = 'balanced')
    
    SVM_Classifier.fit(X_train_N,Y_train)

    y_pred=SVM_Classifier.predict(X_test_N)
    score_test = SVM_Classifier.decision_function(X_test_N) 
    score_train = SVM_Classifier.decision_function(X_train_N) 
    #keyFold, y_real_total, y_pred_total, ids_test, best_parameters_dict, score_pred_total, normalizer, model
    results_object.AddResultsInFold(keyFold, Y_test, y_pred, ids_test, score_test, ids_train, score_train, {'Best C': best_C, 'Best G': best_G}, normalize_transfor, SVM_Classifier)
    

    return results_object


def SimpleClassifier_LR(X_train, Y_train, ids_train, X_test, Y_test, ids_test, number_folds_validation, results_object, keyFold, kernel_linear=False, n_job=-1):
    normalize_transfor=preprocessing.StandardScaler().fit(X_train)
    X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)
    
    C_array =[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 10000]#np.geomspace(1e-4,1e3,8)
    # G_array=np.geomspace(1e-6,1e6,13)
    
    #Best parameters and testSet

    model_linear = linear_model.LogisticRegression(solver = 'lbfgs')

    
    parameters = { 'C':C_array }
    model_linear = linear_model.LogisticRegression(solver = 'liblinear', class_weight = 'balanced')
    clf = model_selection.GridSearchCV(model_linear, parameters, cv=number_folds_validation, n_jobs = n_job)
    clf.fit(X_train_N,Y_train)                
    best_C=clf.best_estimator_.C
    best_penalty = clf.best_estimator_.penalty
    
    #print('####'+str(clf.best_estimator_.kernel)+'#####')

    LR_Classifier = linear_model.LogisticRegression(penalty = best_penalty , C=best_C, class_weight = 'balanced')
    
    LR_Classifier.fit(X_train_N,Y_train)

    y_pred = LR_Classifier.predict(X_test_N)
    score_test = LR_Classifier.decision_function(X_test_N)
    score_train = LR_Classifier.decision_function(X_train_N)
    
    
    results_object.AddResultsInFold(keyFold, Y_test, y_pred, ids_test, score_test, ids_train, score_train, {'Best C': best_C, 'Best penalty': best_penalty}, normalize_transfor, LR_Classifier)
    

    return results_object


def SimpleClassifier_XGBoost(X_train, Y_train, ids_train,
                             X_test, Y_test, ids_test, number_folds_validation,
                             results_object, keyFold, n_job=-1):

    normalize_transfor=preprocessing.StandardScaler().fit(X_train)
    X_train_N,X_test_N=normalize_transfor.transform(X_train),normalize_transfor.transform(X_test)    
    
    learning_rate = [0.001, 0.01, 0.1, 0.3]
    n_estimators = [50, 100, 200, 300]
    max_depth = [1, 3, 5, 7]
    

    
    parameters = {'learning_rate':learning_rate, 'n_estimators':n_estimators, 'max_depth': max_depth}
    xgb_model = xgb.XGBClassifier(objective="binary:logistic")
    clf = model_selection.GridSearchCV(xgb_model, parameters, cv=number_folds_validation, n_jobs = n_job)
    clf.fit(X_train_N,Y_train)
    lr_ideal=clf.best_estimator_.learning_rate
    estimators_ideal=clf.best_estimator_.n_estimators
    max_depth_ideal=clf.best_estimator_.max_depth

    xgb_classifier = xgb.XGBClassifier(objective="binary:logistic", learning_rate=lr_ideal,
                                  n_estimators=estimators_ideal, max_depth=max_depth_ideal)    
    
    xgb_classifier.fit(X_train_N, Y_train)
    y_pred = xgb_classifier.predict(X_test_N)
    score_test = xgb_classifier.predict_proba(X_test)[:,1]
    
    parameter_dict = {'learning_rate': lr_ideal,
                      'n_estimators': estimators_ideal,
                      'max_depth': max_depth_ideal}

    results_object.AddResultsInFold(keyFold, Y_test, y_pred, ids_test, parameter_dict, score_test, normalize_transfor, xgb_classifier)
    

    return results_object
    
    


