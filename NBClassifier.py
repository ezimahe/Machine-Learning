# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:52:45 2019

@author: Prashanti
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

admitData = pd.read_csv("/Users/eleanorezimah/Desktop/AIT 590/Experimental Project/training_data.csv")

features = admitData[["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]]

targetVariables = admitData.salary


finalTargetVariables = []
for x in targetVariables:
    if x <= 0.5:
        finalTargetVariables.append(0)
    else:
            finalTargetVariables.append(1)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, finalTargetVariables, test_size=0.4)

model = BernoulliNB()
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)

print(accuracy_score(targetTest, predictions))

