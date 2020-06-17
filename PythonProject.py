#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:15:53 2019

@author: eleanorezimah
"""

import csv
import pandas as pd
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tabulate import tabulate
import matplotlib.pyplot as plt
import re

data = pd.read_csv(r"/Users/eleanorezimah/Desktop/AIT 590/Experimental Project/training_data.csv")

df = pd.DataFrame(data)
# print(df)

df.replace(' ?', numpy.nan, regex=False, inplace=True)
print(df.columns[df.isna().any()])

#print(tabulate(df, tablefmt='psql'))

# print(df.mean())

df=df.fillna(df.mean())
# print(tabulate(df, tablefmt='psql'))

print(df['workclass'].describe())
common_value = 'Private'
df['workclass'] = df['workclass'].fillna(common_value)
# print(tabulate(df, tablefmt='psql'))

print(df['occupation'].describe())
common_value = 'Prof-speciality'
df['occupation'] = df['occupation'].fillna(common_value)
# print(tabulate(df, tablefmt='psql'))

print(df['native-country'].describe())
common_value = 'United-States'
df['native-country'] = df['native-country'].fillna(common_value)
# print(tabulate(df, tablefmt='psql'))
print(type(df['salary']))
# numpy.isnan(df)
df['sex'].replace([' Male', ' Female'], [1, 0], regex=False, inplace=True)
print(df['sex'])

#df.salary.replace([' <=50K', ' >50K'], [1, 0], inplace=True)


# print(tabulate(df, tablefmt='psql'))

# ....data clean: education num........
# data can be train or test
# var name is variable name: should be passed as strings within ('')
# bins is list of numeric values like [0,6,10,11]
# group names is list of groups you want to create in list form
def bin_var(data, var, bins, group_names):
    bin_value = bins
    group = group_names
    data[var + 'Cat'] = pd.cut(df[var], bin_value, labels=group)


bin_var(df, 'education-num', [0, 6, 11, 16], [0, 1, 2])
# bin_var(test, 'Education Num', [0,6,11,16], ['Low', 'Medium', 'High'])
print(pd.crosstab(df['education-numCat'], df['salary']))
df['education-numCat'] = df['education-numCat'].astype('int64')

# ..............data clean:hours/week.......
bin_var(df, 'hours-per-week', [0, 35, 40, 60, 100], [0, 1, 2, 3])
# bin_var(test, 'Hours/Week', [0,35,40,60,100], ['Low', 'Medium', 'High','VeryHigh'])
print(pd.crosstab(df['hours-per-weekCat'], df['salary'], margins=True))
df['hours-per-weekCat'] = df['hours-per-weekCat'].astype('int64')


# ...........data clean:occupation..........
def occup(x):
    if re.search('managerial', x):
        return 1
    elif re.search('specialty', x):
        return 1
    else:
        return 0


df['occupationCat'] = df.occupation.apply(lambda x: x.strip()).apply(lambda x: occup(x))
# df['occupation']=df.occupation.apply(lambda x: x.strip()).apply(lambda x: occup(x))
print(df['occupationCat'].value_counts())

# ................age............
bin_var(df, 'age', [17, 30, 55, 100], [0, 1, 2])
df['ageCat'] = df['ageCat'].astype('int64')

# ...............marital status.........
df['marital-statusCat'] = df['marital-status'].apply(lambda x: 1 if x.startswith('Married', 1) else 0)

# ....................race................
print(pd.crosstab(df['race'], df['salary'], margins=True))
df['Race_cat'] = df['race'].apply(lambda x: x.strip())
df['Race_cat'] = df['Race_cat'].apply(lambda x: 1 if x == 'White' else 0)


# ...............
def workclas(x):
    if re.search('Private', x):
        return 0
    elif re.search('Self', x):
        return 1
    elif re.search('gov', x):
        return 2
    else:
        return 3


df['WorfClass_cat'] = df.workclass.apply(lambda x: x.strip()).apply(lambda x: workclas(x))
df['WorfClass_cat'] = df.workclass.apply(lambda x: x.strip()).apply(lambda x: workclas(x))
df['WorfClass_cat'].value_counts()

print(tabulate(df.head(15), tablefmt='psql', headers='keys'))

#......................................test  data........................................................

data_test = pd.read_csv(r"C:\Users\nikita\Desktop\data_analytics_material\AIT590\python presentation\ML_test_data_without_output.csv")

df_test = pd.DataFrame(data_test)
# print(df)

df_test.replace(' ?', numpy.nan, regex=False, inplace=True)
print(df_test.columns[df_test.isna().any()])

#print(tabulate(df, tablefmt='psql'))

# print(df.mean())

df_test=df_test.fillna(df.mean())
# print(tabulate(df, tablefmt='psql'))

print(df_test['workclass'].describe())
common_value = 'Private'
df_test['workclass'] = df_test['workclass'].fillna(common_value)
# print(tabulate(df, tablefmt='psql'))

print(df_test['occupation'].describe())
common_value = 'Prof-speciality'
df_test['occupation'] = df_test['occupation'].fillna(common_value)
# print(tabulate(df, tablefmt='psql'))

print(df_test['native-country'].describe())
common_value = 'United-States'
df_test['native-country'] = df_test['native-country'].fillna(common_value)
# print(tabulate(df, tablefmt='psql'))
#print(type(df_['salary']))
# numpy.isnan(df)
df_test['sex'].replace([' Male', ' Female'], [1, 0], regex=False, inplace=True)
print(df_test['sex'])

#df.salary.replace([' <=50K', ' >50K'], [1, 0], inplace=True)


# print(tabulate(df, tablefmt='psql'))

# ....data clean: education num........
# data can be train or test
# var name is variable name: should be passed as strings within ('')
# bins is list of numeric values like [0,6,10,11]
# group names is list of groups you want to create in list form
def bin_var(data, var, bins, group_names):
    bin_value = bins
    group = group_names
    data[var + 'Cat'] = pd.cut(df[var], bin_value, labels=group)


bin_var(df_test, 'education-num', [0, 6, 11, 16], [0, 1, 2])
# bin_var(test, 'Education Num', [0,6,11,16], ['Low', 'Medium', 'High'])
#print(pd.crosstab(df['education-numCat'], df['salary']))
df['education-numCat'] = df['education-numCat'].astype('int64')

# ..............data clean:hours/week.......
bin_var(df_test, 'hours-per-week', [0, 35, 40, 60, 100], [0, 1, 2, 3])
# bin_var(test, 'Hours/Week', [0,35,40,60,100], ['Low', 'Medium', 'High','VeryHigh'])
#print(pd.crosstab(df['hours-per-weekCat'], df['salary'], margins=True))
df_test['hours-per-weekCat'] = df_test['hours-per-weekCat'].astype('int64')


# ...........data clean:occupation..........
def occup(x):
    if re.search('managerial', x):
        return 1
    elif re.search('specialty', x):
        return 1
    else:
        return 0


df_test['occupationCat'] = df.occupation.apply(lambda x: x.strip()).apply(lambda x: occup(x))
# df['occupation']=df.occupation.apply(lambda x: x.strip()).apply(lambda x: occup(x))
print(df_test['occupationCat'].value_counts())

# ................age............
bin_var(df_test, 'age', [17, 30, 55, 100], [0, 1, 2])
df_test['ageCat'] = df_test['ageCat'].astype('int64')

# ...............marital status.........
df_test['marital-statusCat'] = df_test['marital-status'].apply(lambda x: 1 if x.startswith('Married', 1) else 0)

# ....................race................
#print(pd.crosstab(df_test['race'], df['salary'], margins=True))
df_test['Race_cat'] = df_test['race'].apply(lambda x: x.strip())
df_test['Race_cat'] = df_test['Race_cat'].apply(lambda x: 1 if x == 'White' else 0)


# ...............
def workclas(x):
    if re.search('Private', x):
        return 0
    elif re.search('Self', x):
        return 1
    elif re.search('gov', x):
        return 2
    else:
        return 3


df_test['WorfClass_cat'] = df_test.workclass.apply(lambda x: x.strip()).apply(lambda x: workclas(x))
df_test['WorfClass_cat'] = df_test.workclass.apply(lambda x: x.strip()).apply(lambda x: workclas(x))
df_test['WorfClass_cat'].value_counts()

print(tabulate(df.head(15), tablefmt='psql', headers='keys'))

#..........................................................................................................

features = df[['WorfClass_cat','education-numCat', 'ageCat', 'Race_cat',
'hours-per-weekCat',
 'marital-statusCat',
 'occupationCat',
  'sex']]
targetVariables = df.salary

#featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=0.20)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables)
#
featureTest=df_test[['WorfClass_cat','education-numCat', 'ageCat', 'Race_cat',
'hours-per-weekCat',
 'marital-statusCat',
 'occupationCat',
  'sex']]




#................................................................................................................
from openpyxl.workbook import Workbook
model= DecisionTreeClassifier(criterion="entropy",max_depth=15)
clf=model.fit(featureTrain, targetTrain)
predictions = clf.predict(featureTest)
prediction=pd.DataFrame(predictions)
print("Printing csv dataframe")
print(tabulate(prediction, tablefmt='psql'))
#prediction.to_csv(open("prediction.csv", "w"), sep=",")
#prediction.to_excel("prediction_new.xlsx")
#prediction = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')
#print(predictions).to_csv('predictions.csv')


# print(accuracy_score(targetTest, predictions))
# print(confusion_matrix(targetTest, predictions))
# print(classification_report(targetTest, predictions))

