import os
import numpy as np

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score,roc_auc_score,confusion_matrix,classification_report,matthews_corrcoef,precision_recall_fscore_support

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.metrics import roc_curve, f1_score
from sklearn import manifold
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def Spe(test,pred):
    m=confusion_matrix(test,pred)
    tn=m[0][0]
    n=m[0][0]+m[0][1]
    x=tn/n
    return x


def Baseline(x, y, k=2, split_list=[0.2, 0.8], time=5, show_train=True, shuffle=True,cls="svm"):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    for split in split_list:
        print("split", split)
        ss = split
        split = int(x.shape[0] * split)  # as train
        micro_list = []
        macro_list = []
        auc = 0
        pre = 0
        mcc = 0
        acc = 0
        sen = 0
        spe = 0
        f1 = 0
        if time:
            if shuffle:
                permutation = np.random.permutation(x.shape[0])
                x = x[permutation, :]
                y = y[permutation]
            for i in range(time):
                train_x = x[:split, :]
                test_x = x[split:, :]

                train_y = y[:split]
                test_y = y[split:]
                if cls=="svm":
                    estimator =SVC()
                if cls=="rf":
                    estimator=RandomForestClassifier()
                if cls=="gbdt":
                    estimator=GradientBoostingClassifier()
                if cls=="mlp":
                    estimator=MLPClassifier()
                estimator.fit(train_x, train_y)
                y_pred = estimator.predict(test_x)
                f1_macro = f1_score(test_y, y_pred, average='macro')
                f1_micro = f1_score(test_y, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)

                prec, sent, f1t, _ = precision_recall_fscore_support(test_y, y_pred, average="binary")
                f1 += f1t
                sen += sent
                pre = pre + prec
                spe += Spe(test_y, y_pred)

                acct = accuracy_score(test_y, y_pred)
                acc += acct
                mcct = matthews_corrcoef(test_y, y_pred)
                mcc = mcc + mcct
                auc_t = roc_auc_score(test_y, y_pred)
                auc += auc_t

                fpr, tpr, thresholds = roc_curve(test_y, y_pred)

                x = np.vstack((test_x, train_x))
                y = np.append(test_y, train_y)

            print("f1", f1 / time)
            print("pre", pre / time)
            print("auc", auc / time)
            print("mcc", mcc / time)
            print("spe", spe / time)
            print("acc", acc / time)
            print("sen", sen / time)


    return estimator

def my_KNN(x, y, k=2, split_list=[0.2, 0.8], time=5, show_train=True, shuffle=True):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    for split in split_list:
        print("split",split)
        ss = split
        split = int(x.shape[0] * split)#as train
        micro_list = []
        macro_list = []
        auc = 0
        pre = 0
        mcc = 0
        acc = 0
        sen = 0
        spe = 0
        f1 =0
        if time:
            if shuffle:
                permutation = np.random.permutation(x.shape[0])
                x = x[permutation, :]
                y = y[permutation]
            for i in range(time):
                
                # x_true = np.array(x_true)
                train_x = x[:split, :]
                test_x = x[split:, :]

                train_y = y[:split]
                test_y = y[split:]

                estimator = KNeighborsClassifier(n_neighbors=k)
                estimator.fit(train_x, train_y)
                y_pred = estimator.predict(test_x)

                f1_macro = f1_score(test_y, y_pred, average='macro')
                f1_micro = f1_score(test_y, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)

                prec,sent,f1t,_ = precision_recall_fscore_support(test_y,y_pred,average="binary")
                f1+=f1t
                sen+=sent
                pre = pre+prec
                spe+=Spe(test_y, y_pred)

                acct = accuracy_score(test_y,y_pred)
                acc+= acct
                mcct = matthews_corrcoef(test_y,y_pred)
                mcc=mcc+mcct
                auc_t = roc_auc_score(test_y,y_pred)
                auc += auc_t

                fpr , tpr , thresholds = roc_curve(test_y,y_pred)

                x=np.vstack((test_x, train_x))
                y=np.append(test_y, train_y)
            print("f1",f1/time)
            print("pre",pre/time)
            print("auc",auc/time)
            print("mcc",mcc/time)
            print("spe",spe/time)
            print("acc",acc/time)
            print("sen",sen/time)

    return estimator
                
def my_Kmeans(x, y, k=2, time=10, return_NMI=False):

    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    estimator = KMeans(n_clusters=k)
    ARI_list = []  # adjusted_rand_score(
    NMI_list = []
    if time:
        for i in range(time):
            estimator.fit(x, y)
            y_pred = estimator.predict(x)
            score = normalized_mutual_info_score(y, y_pred)
            NMI_list.append(score)
            s2 = adjusted_rand_score(y, y_pred)
            ARI_list.append(s2)

        score = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        print('NMI (10 avg): {:.4f} , ARI (10avg): {:.4f}'.format(score, s2))

    else:
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return score, s2
