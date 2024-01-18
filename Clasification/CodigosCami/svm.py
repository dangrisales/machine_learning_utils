import sys
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as st
from scipy import interp
import itertools

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, recall_score, roc_curve, auc, classification_report, f1_score
from sklearn.utils import class_weight
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from stratkfold import stratified_group_k_fold
from sklearn.decomposition import PCA
#np.set_printoptions(threshold=sys.maxsize)

def read_feat_mat(file_feat_csv, label):

    feat=pd.read_csv(file_feat_csv, sep=";")
    print(feat.head())
    keys=feat.keys().tolist()
    metadata_keys=["ID", "Unnamed: 0", "class",  "class2",  "gender",  "age",  "scholar",  "moca", "ineco",  "hy", "updrs3", "Unnamed: 0.1"]
    for k in metadata_keys:
        if k in keys:
            keys.remove(k)
    if label==0:
        y=feat["class"].values
        ID=np.asarray([feat["ID"][k][0:5] for k in range(len(feat["ID"]))])
        featmat=np.stack([feat[k] for k in keys], axis=1)
        if type(featmat[0,0])==str:
            for r in range(featmat.shape[0]):
                for c in range(featmat.shape[1]):
                    featmat[r,c]=float(featmat[r,c].replace(',','.'))
        featmat = featmat.astype(np.float)
        return featmat, ID, y, keys
    elif label==1:
        y1="CTR_DCL"
        y2="Park_DCL"
    elif label==2:
        y1="CTR_noDCL"
        y2="Park_noDCL"        
    elif label==3:
        y1="Park_noDCL"       
        y2="Park_DCL"

    feat1=feat[(feat["class2"]==y1) | (feat["class2"]==y2)]
    yt=feat1["class2"].values

    y=np.asarray([0 if yt[j]==y1 else 1 for j in np.arange(len(yt))])
    IDV=feat1["ID"].values
    ID=np.asarray([IDV[k][0:5] for k in range(len(IDV))])
    featmat=np.stack([feat1[k].replace(',', '.') for k in keys], axis=1)
    if type(featmat[0,0])==str:
        for r in range(featmat.shape[0]):
            for c in range(featmat.shape[1]):
                featmat[r,c]=float(featmat[r,c].replace(',','.'))
                
    featmat = featmat.astype(np.float)
    return featmat, ID, y, keys


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          cmap=plt.cm.Blues, file_save="./cm"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(4,4))
    cm = confusion_matrix(y_true, y_pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    
    plt.ylabel('Real class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.savefig(file_save+".png")
    plt.savefig(file_save+".pdf")


def plotPCAsemantic(X,feat_names, nComp=0.8, top=10, file_save="pca"):

    print(feat_names)
    pca=PCA(n_components=nComp)

    Xpca=pca.fit_transform(X)


    Xcomp=np.abs(pca.components_)

    Xsum=np.sum(Xcomp, 0)

    pos_feat=np.argsort(Xsum)[::-1]

    print(pos_feat)
    top_feat=feat_names[pos_feat][0:top]
    top_importance=Xsum[pos_feat][0:top]

    ax = sns.barplot(top_feat, y=top_importance, palette="Blues_d")

    for item in ax.get_xticklabels():
        item.set_rotation(60)

    plt.ylabel("Feature importance")
    plt.tight_layout()
    plt.savefig(file_save+".png")
    plt.savefig(file_save+".pdf")
    plt.close()


def static_class_score(ID, score, y_pred, y):

    IDs=np.unique(ID)
    print(IDs)
    y_sp_pred=np.zeros(len(IDs))
    y_sp=np.zeros(len(IDs))
    score_sp=np.zeros(len(IDs))
    for j in range(len(IDs)):
        ny_pred=np.where(ID==IDs[j])[0]
        pred_sp=y_pred[ny_pred]
        real_sp=y[ny_pred]
        sc_sp=score[ny_pred]
        y_sp_pred[j]=st.mode(pred_sp)[0]
        y_sp[j]=st.mode(real_sp)[0]
        score_sp[j]=np.mean(sc_sp)
        #print(pred_sp, real_sp )
    return np.asarray(y_sp_pred), np.asarray(y_sp), np.asarray(score_sp)

if __name__ == "__main__":

    NFOLDS=10

    if len(sys.argv)!=4:
        print("python svm.py <filefeat> <file_results> <class>")
        sys.exit()
    class_p=int(sys.argv[3])
    X, person, y, feat_names=read_feat_mat(sys.argv[1], class_p)
    file_results=sys.argv[2]
    doPCA=False

    X[np.isinf(X)]=0
    X[np.isnan(X)]=0

    keep=[]
    for k in range(X.shape[0]):
        if np.sum(X[k,:])!=0:
            keep.append(k)
    keep=np.hstack(keep)

    X=X[keep,:]

    y=y[keep]
    person=person[keep]
    print(X.shape)

    acc=np.zeros(NFOLDS)
    uar=np.zeros(NFOLDS)
    uardev=np.zeros(NFOLDS)

    fscore=np.zeros(NFOLDS)
    sens=np.zeros(NFOLDS)
    spec=np.zeros(NFOLDS)
    aucs=np.zeros(NFOLDS)
    C=np.zeros(NFOLDS)
    gamma=np.zeros(NFOLDS)


    j=0
    y_pred_all=[]
    y_test_all=[]
    y_score=[]

    id_all_test=[]
    id_fold=[]


    scalerT=StandardScaler()
    Xe=scalerT.fit_transform(X)
    X_embedded = TSNE(n_components=2).fit_transform(Xe)
    if class_p==0:
        label_s=["HC" if j==0 else "PD" for j in y]
    elif class_p==1:
        label_s=["CTR_DCL" if j==0 else "Park_DCL" for j in y]
    elif class_p==2:
        label_s=["CTR_noDCL" if j==0 else "Park_noDCL" for j in y]   
    elif class_p==3:
        label_s=["Park_noDCL" if j==0 else "Park_DCL" for j in y]


    # plotPCAsemantic(X,np.asarray(feat_names), file_save=file_results+"_PCA")


    # sys.exit()


    # plt.figure(figsize=(4,4))
    # ax = sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=label_s, palette=sns.color_palette("cubehelix", 2), alpha=0.7, s=50)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(file_results+"TSNE.png")
    # plt.savefig(file_results+"TSNE.pdf")

    parameters = {'kernel':['rbf'], 'class_weight': ['balanced'],
            'C':st.expon(scale=100),
            'gamma':st.expon(scale=0.001)}

    # parameters = {'kernel':['rbf'],
    #         'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #         'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    

    ## SVM optimization process. Get the optimal C and gamma parameters

    for train_index, test_index in stratified_group_k_fold(X, y, person, k=NFOLDS, seed=42):
        start=datetime.now()
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        id_all_test=person[test_index]

        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)


        if doPCA:
            pca = PCA(n_components=0.8)
            X_train=pca.fit_transform(X_train)
            X_test=pca.transform(X_test)


        svc = svm.SVC()

        clf=RandomizedSearchCV(svc, parameters, n_jobs=4, cv=NFOLDS-1, verbose=1, n_iter=500, scoring='balanced_accuracy')

        clf.fit(X_train, y_train)
        print("--------------- Best Params ------------------")
        print(clf.best_params_)
        print(clf.best_score_)
        C[j]=clf.best_params_['C']
        gamma[j]=clf.best_params_['gamma']
        uardev[j]=clf.best_score_
        j=j+1

    print(C, gamma)
    Copt=np.median(C)
    gammaopt=np.median(gamma)
    print(Copt, gammaopt)
    mean_fpr = np.linspace(0, 1, 100)

    ## Fit the SVM with the optimal parameters
    j=0
    tprsm=[]
    for train_index, test_index in stratified_group_k_fold(X, y, person, k=NFOLDS, seed=42):
        start=datetime.now()
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        id_all_test=person[test_index]


        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)

        if doPCA:

            pca = PCA(n_components=0.8)
            X_train=pca.fit_transform(X_train)
            X_test=pca.transform(X_test)

        svc = svm.SVC(C=Copt, gamma=gammaopt, kernel="rbf", class_weight= 'balanced')
        svc.fit(X_train, y_train)

        y_predv=svc.predict(X_test)

        score=svc.decision_function(X_test)
        
        ypred_sp, yreal_sp, score_sp=static_class_score(id_all_test, score, y_predv, y_test)
        y_score.append(score_sp)
        y_test_all.append(yreal_sp)
        y_pred_all.append(ypred_sp)
        id_fold.append(np.unique(id_all_test))        

        acc[j]=accuracy_score(yreal_sp, ypred_sp)
        uar[j]=balanced_accuracy_score(yreal_sp, ypred_sp)

        fscore[j]=f1_score(yreal_sp, ypred_sp, average="macro")
        [spec[j], sens[j]]=recall_score(yreal_sp, ypred_sp, average=None)

        fprs, tprs, thresholds = roc_curve(yreal_sp, score_sp)
        tprsm.append(interp(mean_fpr, fprs, tprs))
        aucs[j] = auc(fprs, tprs)
        tprsm[-1][0] = 0.0
        print(acc)
        print(uar)
        print(spec)
        print(sens)
        print(aucs)
        j+=1


    y_real=np.hstack(y_test_all)
    y_pred=np.hstack(y_pred_all)
    score_val=np.hstack(y_score)
    dfclass=classification_report(y_real, y_pred,digits=4)

    [specF, sensF]=recall_score(y_real, y_pred, average=None)
    fprF, tprF, thresholds = roc_curve(y_real, score_val)
    roc_aucF = auc(fprF, tprF)
    print(dfclass)


    np.set_printoptions(precision=2)

    if class_p==0:
        l1="HC"
        l2="PD"
    elif class_p==1:
        l1="CTR_DCL"
        l2="Park_DCL"
    elif class_p==2:
        l1="CTR_noDCL"
        l2="Park_noDCL"        
    elif class_p==3:
        l1="Park_noDCL"       
        l2="Park_DCL"

    plot_confusion_matrix(y_real, y_pred, classes=[l1, l2], normalize=False, file_save=file_results+"CMu")
    plot_confusion_matrix(y_real, y_pred, classes=[l1, l2], normalize=True, file_save=file_results+"CMn")


    plt.figure(figsize=(4,4))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprsm, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(fprF, tprF, color='k',
         label=r'Avg. ROC (AUC = %0.3f $\pm$ %0.3f)' % (roc_aucF, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprsm, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(file_results+"AUC.png")
    plt.savefig(file_results+"AUC.pdf")

 


    F=open(file_results+".csv", "a")
    header="fold,UAR_dev,ACC_test,Fscore,UAR_SP,SENS,SPEC,AUC,C,gamma\n"
    F.write(header)
    for j in range(len(acc)):
        content=str(j+1)+","
        content+=str(uardev[j])+","
        content+=str(acc[j])+","
        content+=str(fscore[j])+","
        content+=str(uar[j])+","
        content+=str(sens[j])+","
        content+=str(spec[j])+","
        content+=str(aucs[j])+","
        content+=str(C[j])+","
        content+=str(gamma[j])+"\n"
        F.write(content)
    
    content="AVG."+","
    content+=str(np.mean(uardev))+","
    content+=str(np.mean(acc))+","
    content+=str(np.mean(fscore))+","
    content+=str(np.mean(uar))+","
    content+=str(np.mean(sens))+","
    content+=str(np.mean(spec))+","
    content+=str(np.mean(np.asarray(aucs)))+"_"+str(roc_aucF)+","
    content+=str(Copt)+","
    content+=str(gammaopt)+"\n"
    F.write(content)
    content="STD."+","
    content+=str(np.std(uardev))+","
    content+=str(np.std(acc))+","
    content+=str(np.std(fscore))+","
    content+=str(np.std(uar))+","
    content+=str(np.std(sens))+","
    content+=str(np.std(spec))+","
    content+=str(np.std(np.asarray(aucs)))+",,,\n"
    F.write(content)
    F.close()


    df={'ID':list(itertools.chain.from_iterable(id_fold)), 'class': np.hstack(y_real), 'pred_class': np.hstack(y_pred), 'score': np.hstack(score_val)}

    df=pd.DataFrame(df)
    df.to_csv(file_results+"per_spk.csv")

    plt.figure(figsize=(4,4))
    p1=np.where(np.hstack(y_real)==0)[0]
    p2=np.where(np.hstack(y_real)==1)[0]
    sns.distplot(np.hstack(score_val)[p1], label=l1)
    sns.distplot(np.hstack(score_val)[p2], label=l2, color="k")
    plt.grid()
    plt.legend()
    plt.xlabel("Classification threshold")
    plt.ylabel('Normalized count')
    plt.tight_layout()
    plt.savefig(file_results+"Histogram.png")
    plt.savefig(file_results+"Histogram.pdf")
    #plt.show()








