#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:12:06 2020

@author: Tanmay Basu

"""


import numpy as np
from scipy.spatial import distance


class TkNN():
    def __init__(self,theta = 0.25,beta=2,metric='cosine'):
        self.theta = theta
        self.beta = beta
        self.metric= metric

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.class_names = np.unique(self.y_train)

    def TkNN(self,x_test):
        L = self.beta
        S = []
        class_count = np.empty(self.class_names.shape[0])
        for x_train in self.X_train:
#            st=globals()["distance."+self.metric]
#            dist=st(x_train,x_test)
            if self.metric=='cosine':
                dist = distance.cosine(x_train,x_test) 
            elif self.metric=='chebyshev':
                dist = distance.chebyshev(x_train,x_test) 
            elif self.metric=='cityblock':
                dist = distance.cityblock(x_train,x_test) 
            elif self.metric=='euclidean':
                dist = distance.euclidean(x_train,x_test) 
            elif self.metric=='minkowski':
                dist = distance.minkowski(x_train,x_test) 
            else:
                print('Error!!! Enter a correct distance function and try again \n')
            if dist > self.theta:
                S.append(dist)
        SPN_index = np.argsort(S)      # Returning the indices of the similarity values (distance values are sorted in ascending order)   
        SPN_labels = np.empty(SPN_index.shape[0]) 
        for i in range(SPN_index.shape[0]):
            SPN_labels[i] = self.y_train[SPN_index[i]]
        SPN_0_labels = SPN_labels[:L]
        for i in range(self.class_names.shape[0]):
            val = 0
            for j in range(SPN_0_labels.shape[0]):
                if SPN_0_labels[j] == self.class_names[i]:
                    val = val + 1
            class_count[i] = val
        while L <= SPN_labels.shape[0]:
            L_x1 = np.max(class_count)
            L_x1_class = np.argmax(class_count)
            L_x2 = np.max(np.delete(class_count,np.argmax(class_count)))
            if L_x1 - L_x2 == self.beta:
                return L_x1_class
            else:
                L = L + 1
                if L==SPN_labels.shape[0]:        # Checked the last element of the training set 
                    return -1
                else:
                    c = SPN_labels[L]
                    class_count[int(c)] += 1 
        return -1     
    
    def predict(self,X_test):
        Y_test = np.empty(X_test.shape[0])
        for i in range(X_test.shape[0]):
            Y_test[i]=self.TkNN(X_test[i])
        return Y_test
    


   