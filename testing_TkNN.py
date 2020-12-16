#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:12:06 2020

@author: Tanmay Basu

"""
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csc_matrix
from collections import Counter
from sklearn.datasets import load_iris
from TkNN import TkNN


##########################   Lodaing IRIS Data  ##########################  
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


##########################  Classification ##########################  
try:
    theta=float(input("Enter the value of theta OR press Enter, if you want to use the default value: "))
    beta=int(input("Enter the value of beta OR press Enter, if you want to use the default value: "))
    distance=input("Enter the name of the metric: ")  
except:  
    theta = 0.25; beta=2; distance='cosine';
    print('The default values of theta and beta will be used \n')
clf=TkNN(theta,beta,distance)    
clf.fit(X_train,y_train)
predicted_class_label = clf.predict(X_test)

# Evaluation
print ('Evaluation using Precision, Recall and F-measure (macro)')
pr=precision_score(y_test, predicted_class_label, average='macro')
print ('\n Precision:'+str(pr))
re=recall_score(y_test, predicted_class_label, average='macro')
print ('\n Recall:'+str(re))
fm=f1_score(y_test, predicted_class_label, average='macro') 
print ('\n Macro Averaged F1-Score :'+str(fm))
fm=f1_score(y_test, predicted_class_label, average='micro') 
print ('\n Mircro Averaged F1-Score:'+str(fm))



