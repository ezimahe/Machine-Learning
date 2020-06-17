#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:17:53 2019

@author: eleanorezimah
"""


#####################Import the packages 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sn
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


####################IMPORT THE DATABASE

columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']
train = pd.read_csv('/Users/eleanorezimah/Desktop/AIT 590/Experimental Project/training_data.csv', names=columns)
test = pd.read_csv('/Users/eleanorezimah/Desktop/AIT 590/Experimental Project/ML_test_data_without_output.csv', names=columns, skiprows=1)
train.info()


####################Clean the Data

df = pd.concat([train, test], axis=0)
dff=df
k=df

df['Income'] = df['Income'].apply(lambda x: 1 if x==' >50K' else 0)

for col in df.columns:
    if type(df[col][0]) == str:
        print("Working on " + col)
        df[col] = df[col].apply(lambda val: val.replace(" ",""))


####################REMOVE UNKNOWNS
    
df.replace(' ?', np.nan, inplace=True)###making copy for visualization


#################### Converting to int

df = pd.concat([df, pd.get_dummies(df['Workclass'],prefix='Workclass',prefix_sep=':')], axis=1)
df.drop('Workclass',axis=1,inplace=True)

df = pd.concat([df, pd.get_dummies(df['Marital Status'],prefix='Marital Status',prefix_sep=':')], axis=1)
df.drop('Marital Status',axis=1,inplace=True)

df = pd.concat([df, pd.get_dummies(df['Occupation'],prefix='Occupation',prefix_sep=':')], axis=1)
df.drop('Occupation',axis=1,inplace=True)

df = pd.concat([df, pd.get_dummies(df['Relationship'],prefix='Relationship',prefix_sep=':')], axis=1)
df.drop('Relationship',axis=1,inplace=True)

df = pd.concat([df, pd.get_dummies(df['Race'],prefix='Race',prefix_sep=':')], axis=1)
df.drop('Race',axis=1,inplace=True)

df = pd.concat([df, pd.get_dummies(df['Sex'],prefix='Sex',prefix_sep=':')], axis=1)
df.drop('Sex',axis=1,inplace=True)

df = pd.concat([df, pd.get_dummies(df['Native country'],prefix='Native country',prefix_sep=':')], axis=1)
df.drop('Native country',axis=1,inplace=True)

df.drop('Education', axis=1,inplace=True)

df.head()



######################## Visualizations #############################
########################################### VISULIZATION ##################################################
###########################################################################################################
###########################################################################################################

plt.hist(dff['Age']);

dff['Income'] = dff['Income'].apply(lambda x: 1 if x==' >50K.' else 0)

dff.replace(' ?', np.nan, inplace=True)
#The output for the this line of code can be viewed at https://tinyurl.com/y8ddex6h
###################################  WORKCLASS
dff.fillna(' 0', inplace=True)

sn.factorplot(x="Workclass", y="Income", data=dff, kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=45);
dff['Workclass'].value_counts()
#The output for this line of code can be viewwed at https://tinyurl.com/y9z647j8
########################################### EDUCATION
sn.factorplot(x="Education",y="Income",data=dff,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);
#The output for this line of code can be viewed at https://tinyurl.com/y8wcleoo
#########################  EDUCATION NO
sn.factorplot(x="Education num",y="Income",data=dff,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);
#The output for this line of code can be viewed at https://tinyurl.com/y8zatgbb
################################ MARITAL status
sn.factorplot(x="Marital Status",y="Income",data=dff,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);
#The output for this line of code can be viewed at https://tinyurl.com/ybpggx5u
################################ OCCUPATION
sn.factorplot(x="Occupation",y="Income",data=dff,kind="bar", size = 8, 
palette = "muted")
plt.xticks(rotation=60);
#The output for this line of code can be viewed at https://tinyurl.com/yab83lf3
################################ Relationship
sn.factorplot(x="Relationship",y="Income",data=dff,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);
#The output for this line of code can be viewed at https://tinyurl.com/y9trdq5q
################################ RACE
sn.factorplot(x="Race",y="Income",data=dff,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=45);
#the output for this line of code can be viewed at : https://tinyurl.com/y9hfqlkr
################################ SEX
sn.factorplot(x="Sex",y="Income",data=dff,kind="bar", size = 4, 
palette = "muted");
              
################################     Native county   
sn.factorplot(x="Native country",y="Income",data=dff,kind="bar", size = 10, 
palette = "muted")
plt.xticks(rotation=80);
#Output for this line of code can be viewewd at : https://tinyurl.com/yaqwzr9d





########### Preparing data for Training and testing 

X = np.array(df.drop(['Income'], 1))
y = np.array(df['Income'])
X = preprocessing.scale(X)
y = np.array(df['Income'])

#Splitting data as train and test data 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)



 
############################################## KNN ###############

from sklearn import preprocessing, cross_validation, neighbors
from sklearn.metrics import accuracy_score

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)

knnpre = clf.predict(X_test)

##########Results

print(confusion_matrix(y_test,knnpre))
print(classification_report(y_test,knnpre))
KKNA = accuracy_score(y_test, knnpre)
print("The Accuracy for KNN is {}".format(KKNA))
#The output for this line of code can be found at : https://tinyurl.com/y825rn7v
 


####################### NAIVE  #####################################

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score

model = GaussianNB()

# Train the model using the training sets 
model.fit(X_train, y_train)

#Predict Output 
naive_pre= model.predict(X_test)
print (naive_pre)
##result
print(confusion_matrix(y_test,naive_pre))
print(classification_report(y_test,naive_pre))
NBA = accuracy_score(y_test, naive_pre)
print("The Accuracy for NB is {}".format(NBA))
#The output for this line of code can be found at : https://tinyurl.com/yatb56wa


