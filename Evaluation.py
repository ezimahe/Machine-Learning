#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:45:14 2019

@author: eleanorezimah
"""
# Elijah Bass
#
# Evaluating Classifiers and Regressors Demonstration
#
# GMU AIT 590 Fall 2019
#
# Source:   Jason Brownlee, "Metrics to Evaluate Machine Learning Algorithms in Python"
#           https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
import numpy as np
import csv

# This Section Demonstrates The Following
# Classification Evaluation Metrics:
# 1. Classification Accuracy (Classification Metrics)
# 2. Logarithmic Loss (Classification Metrics)
# 3. Area Under ROC Curve (Classification Metrics)
# 4. Confusion Matrix (Classification Prediction Results)
# 5. Classification Report (Classification Prediction Results)

print("\n\nEvaluating Classifiers Demonstration\n")
# Open Data file and store as list data
data = []
with open("pima-indians-diabetes.data.csv", 'r') as file:
    my_reader = csv.reader(file, delimiter=',')
    for row in my_reader:
        data.append(row)

# Convert all the appended list data to numpy array
dataframe = np.array(data, dtype=np.float64)

# Slice the data by assigning the features from the frist 8 columns
X = dataframe[:,0:8]
# Slice the data by assigning the class from the 9th column
Y = dataframe[:,8]

# Cross Validation Classification Accuracy

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)

# Classification Accuracy
model = LogisticRegression(solver='liblinear')
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %0.3f (%0.3f)" % (results.mean(), results.std()))

# Logarithmic Loss
scoring = 'neg_log_loss'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("LogLoss: %0.3f (%0.3f)" % (results.mean(), results.std()))

# Area Under ROC Curve
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: %0.3f (%0.3f)" % (results.mean(), results.std()))

# Cross Validation Classification Confusion Matrix
test_size = 0.33
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print("Confusion Matrix:")
print(matrix)

# Cross Validation Classification Report
report = classification_report(Y_test, predicted)
print("Classification Report:")
print(report)



# In this section demonstrates Three of the most common metrics for
# Evaluating predictions on regression machine learning problems as follows:
# 1. Mean Absolute Error
# 2. Mean Squared Error
# 3. R^2

print("\n\nEvaluating Regressors Demonstration\n")
# Open Data file and store as list data
data = []
with open("housing.data.csv", 'r') as file:
    my_reader = csv.reader(file, delimiter=',')
    for row in my_reader:
        data.append(row)

# Convert all the appended list data to numpy array
dataframe = np.array(data, dtype=np.float64)


X = dataframe[:,0:13]
Y = dataframe[: , 13]
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearRegression()

# Cross Validation Regression MAE
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))

# Cross Validation Regression MSE
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MSE: %.3f (%.3f)" % (results.mean(), results.std()))

# Cross Validation Regression R^2)
scoring = 'r2'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))
