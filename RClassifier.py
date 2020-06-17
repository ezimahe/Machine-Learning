#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:45:12 2019

@author: eleanorezimah
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 01:59:27 2019

@author: Owner
"""


#from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
le = preprocessing.LabelEncoder()
import csv
# Reading the Specific File with CSV reader library
file=open(r'C:\Users\Owner\Desktop\trainnew1.txt', "r")
reader = csv.reader(file)
data = []
for line in reader:
    # Appending each line of CV files 
    data.append(line)
# Converting all the appended list data to numpy array
data = np.array(data)
# Assigning Column values to cols variable
cols = data[0]
# Removing Column values 
data = data[1:]
# Function to convert the X data value to Float value unless the value is string of ''
def mysub(astr):
    if astr=='':
        return np.nan
    else:
        return np.float32(astr)
f = np.frompyfunc(mysub,1,1)
# Defining the Numeric Features column Expicitely 
NumCols = [0,2,5,6,7,9]
# Picking Object features column list
StrCols = [i for i in range(data.shape[1]) if i not in NumCols]
# Encoding data by applying the label encoder for the object features train data
EncData = np.hstack([f(data[:,NumCols]),np.apply_along_axis(le.fit_transform,1,data[:,StrCols])])
# Converting the Encoded object features data to numpy float data type
EncData = np.array(EncData, dtype=np.float64)
# Picking the Predicting variable from traind data
PredVariable = data[:,1].astype(int)

#Obtain mean of columns as you need, nanmean is just convenient.
col_mean = np.nanmean(EncData, axis=0)
#Find indicies that you need to replace
inds = np.where(np.isnan(EncData))
#Place column means in the indices. Align the arrays using take
EncData[inds] = np.take(col_mean, inds[1])
# Splitting the data to train and test
X_train, X_test, y_train, y_test = train_test_split(EncData, PredVariable, test_size=0.33)
# Defining and applying Ridge aclassifier to data
clf = RidgeClassifier().fit(X_train, y_train)
clf.score(X_train, y_train) 

# Pring the Classification report of predictions
print(classification_report(y_test,clf.predict(X_test)))