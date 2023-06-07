#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang

Source code for the PReNet algorithm in KDD'23.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from joblib import Memory
from sklearn.datasets import load_svmlight_file

mem = Memory("./data/svm_data")

@mem.cache
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]

def dataLoading(path):
    # loading data
    df = pd.read_csv(path) 
    
    labels = df['class']
    
    x_df = df.drop(['class'], axis=1)
    
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    
    return x, labels;


def dataLoading_noheader(path):
    # loading data
    df = pd.read_csv(path, header=None) 
    
        
    x = df.values
#    print("Data shape: (%d, %d)" % x.shape)
    
    return x;

    
def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
#    print(roc_auc)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
#    plt.title('Receiver Operating Characteristic')
#    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
#    plt.legend(loc='lower right')
#    plt.plot([0,1],[0,1],'r--')
#    plt.xlim([-0.001, 1])
#    plt.ylim([0, 1.001])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show();
    return roc_auc, ap;


def writeOutlierScores(scores, labels, name):
    csv_file = open('./outlierscores/' + name + '.csv', 'w') 
#"w" indicates that you're writing strings to the file

    columnTitleRow = 'class,score\n'
    csv_file.write(columnTitleRow)

    for idx in range(0, len(scores)):
        row = str(labels[idx]) + "," + str(scores[idx][0]) + "\n"
        csv_file.write(row)

def writeRepresentation(data, labels, dim, name):
    path = ('./dataset/embedding/' + name + '_rp_' + str(dim) + '.csv') 
#"w" indicates that you're writing strings to the file
    attr_names = [0] * (dim + 1)
    for i in range(0, dim):
        attr_names[i]=  'attr' + str(i)
    
        
    attr_names[dim] = 'class'
    labels = labels.reshape((len(labels), 1))
    print(labels.shape, data.shape)
    data = np.concatenate((data, labels), axis = 1)
    df = pd.DataFrame(data)
    df.to_csv(path, header = attr_names, index=False)

def writeResults(name, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers, depth, rauc, ap, std_auc, std_ap, train_time, test_time, path = "./results/auc_performance_cl0.5.csv"):    
    csv_file = open(path, 'a') 
    row = name + "," + str(n_samples)+ ","  + str(dim) + ',' + str(n_samples_trn) + ','+ str(n_outliers_trn) + ','+ str(n_outliers)  + ',' + str(depth)+ "," + str(rauc) +"," + str(std_auc) + "," + str(ap) +"," + str(std_ap)+"," + str(train_time)+"," + str(test_time) + "\n"
    csv_file.write(row)

def visualizeData(data, labels, name):
    plt.figure(figsize=(5, 5))
    plt.plot(data[labels == 1, 0], data[labels == 1, 1], 'ro')
    plt.plot(data[labels != 1, 0], data[labels != 1, 1], 'bo')
    plt.title('2-D ' + name)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(['outliers', 'inliers'], loc='upper right')
    plt.show()

