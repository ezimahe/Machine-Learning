#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:45:09 2019

@author: eleanorezimah
"""

#Algorithm has 4 steps
#step 1:choose the no of k neighbours
#step 2:take the k nearest neighbour of the new data point according to your distance metric
#step 3:amoung those k neighbours count the number of datapoints to each category
#step 4:Assign the new data point to the category where you counted the most neighbours
#import libraries
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
#load dataset without using pandas
file=open("/Users/eleanorezimah/Desktop/AIT 590/Experimental Project/Iris.txt", "r",encoding='utf-16')
reader = csv.reader(file)
ID = []
SepalLength = []
SepalWidth =[]
PetalLength = []
PetalWidth = []
Labels = []
for line in list(reader)[1:]:
  
    line=line[0].split('\t')
   # print(line)
 
    ID.append(line[0])
    SepalLength.append(line[1])
    SepalWidth.append(line[2])
    PetalLength.append(line[3])
    PetalWidth.append(line[4])
    Labels.append(line[5])
 
#As we can see dataset contain six columns: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm and Species.
# The actual features are described by columns 1-4. Last column contains labels of samples. 
#Firstly we need to split data into two arrays: X (features) and y (labels).
X = np.vstack([SepalLength,SepalWidth,PetalLength,PetalWidth]).astype(np.float16).T
Labels = np.array(Labels)
#Pairplot is useful when you want to visualize the distribution of a variable 
#or the relationship between multiple variables separately within subsets of your dataset.
df=sns.load_dataset("iris")
sns.pairplot(df,hue="species")
#Creating training and test datasets
#Let's split dataset into training set and test set, 
#to check later on whether or not our classifier works correctly.
data_train, data_test, label_train, label_test = train_test_split(X, Labels, test_size=0.30)
#Choosing the best value for k based on the error rate
error_rate = []
kmax=75
#make a list of the k neighbors' targets
for i in range(1,kmax):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(data_train,label_train)
    pred = knn.predict(data_test)
    error_rate.append(np.mean(pred != label_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,kmax),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
# Instantiate learning model (k = 1)
knn = KNeighborsClassifier(n_neighbors=1)
# Fitting the model
knn.fit(data_train,label_train)
# Predicting the Test set results
pred = knn.predict(data_test)
print(classification_report(label_test,pred))

#Calculating model accuracy:
cm=confusion_matrix(label_test,pred)
print(cm)
accuracy = accuracy_score(label_test,pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
#Best value for k
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(data_train,label_train)
predict = knn.predict(data_test)
print(classification_report(label_test,predict))