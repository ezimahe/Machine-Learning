#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# load data
pima = pd.read_csv("titanic1.txt")
print(pima.head())

df= pd.DataFrame(pima)

# check null values
print(df.columns[df.isnull().any()])

# treat missing values
df.replace('', numpy.NaN)
print(df.mean())
df=df.fillna(df.mean())
#print(df[['Age']])


df['Embarked'].describe()
common_value = 'S'
df['Embarked'] = df['Embarked'].fillna(common_value)


df.dtypes
df.Sex.replace(['male', 'female'], [1, 0], inplace=True)

df.Embarked.replace(['S','C','Q'],[0,1,2],inplace=True)
print(df[['Embarked']])


feature_cols=['Pclass','Age','Sex','SibSp','Parch','Fare','Embarked']
X = df[feature_cols] # Features
y = df.Survived# Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) # 75% training and 25% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))

#Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy",max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy1:",accuracy_score(y_test, y_pred))

#performance of classification model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

