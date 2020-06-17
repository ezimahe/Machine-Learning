#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:45:13 2019

@author: eleanorezimah
"""
#Ashwini Python Presentation 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


#Scatter plot outlier analysis
def oplot(df):
    fig, ax = plt.subplots()
    ax.scatter(x = data['GrLivArea'], y = data['SalePrice'])
    plt.xlabel('GrLivArea', fontsize=13)
    plt.show()
    
#Distribution plot for target variable analysis 
def distributionplot(df):
    plt.subplots(figsize=(12,9))
    sns.distplot(data['SalePrice'], bins=100, hist_kws={'alpha': 0.4})
    # Get the fitted parameters used by the function
    (mu, sigma) = stats.norm.fit(data['SalePrice'])
    # plot with the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')

#Filling Nan with random normal distirubtion values
def fillNaN_with_distribution(df):
    a = df.values
    m = np.isnan(a) # mask of NaNs
    mu, sigma = df.mean(), df.std()
    print(mu)
    print(sigma)
    a[m] = np.random.normal(mu, sigma, size=m.sum())
    return df


with open('Housing.txt') as fp:
        #open the file and read it into a list
        mydata = [line.strip().split(",") for line in fp]

#Data manupulation - To perform dataprofiling        
Header = mydata.pop(0)
datan=np.asarray(mydata)
data = pd.DataFrame(datan,columns=Header) 

cols = ['LotArea', 'HouseAge', 'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','SalePrice','NoBedroom']
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce', axis=1)

#Step 1: Data profiling is concerned with summarizing your dataset through descriptive statistics
print(data['SalePrice'].describe())
print(data['GrLivArea'].describe())
print(data['HouseStyle'].describe())

#Step2 : Missing values 
print(data.columns[data.isna().any()])
data = data.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
print(data.columns[data.isna().any()])
print('***************-- Final list')
X=data.values

#Handling missing data - Stragegy 1
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
imputer=imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

dataf = pd.DataFrame(X,columns=Header) 
print('***************')
print(dataf.columns[dataf.isna().any()])
print('***************')

#Handling missing data - Stragegy 2 
dataf['LotArea']=fillNaN_with_distribution(data['LotArea'])
print('***************')
print(dataf.columns[dataf.isna().any()])
print('***************')

#handling missing data - Stragegy 3 
permutation = np.random.permutation(dataf['RoofStyle'])

#erase the empty values
empty_is = np.where(permutation == "")
permutation = np.delete(permutation, empty_is)

#replace all empty values of the dataframe[field]
end = len(permutation)
dataf['RoofStyle'] = dataf['RoofStyle'].apply(lambda x: permutation[np.random.randint(end)] if pd.isnull(x) else x)

print('***************')
print(dataf.columns[dataf.isna().any()])
print('***************')

#Step 2 : Performing Visualization to understand data 
del data['Id']
corr = data.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)
plt.show()

dataf['HouseStyle'].value_counts().plot(kind='bar',title ="VISUALIZATION OF DIFFERENT HOUSESTYLE",figsize=(15,10),legend=True, fontsize=12)
plt.xlabel('House Style')
plt.ylabel('COUNT')
plt.show()

dataplot = data[['LotArea', 'HouseAge', 'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','SalePrice','NoBedroom']].copy()
for i in range(0, len(dataplot.columns), 5):
    sns.pairplot(data=dataplot,
                x_vars=dataplot.columns[i:i+5],
                y_vars=['SalePrice'])

#Step 3: Handling categorical variables
datacategorical = dataf[['HouseStyle', 'RoofStyle', 'BsmtQual','SaleCondition']].copy()
Xa=datacategorical.values
onehotencoder=OneHotEncoder()
Xa = onehotencoder.fit_transform(Xa).toarray()


#Step 4: Feature scaling
numericategory = dataf[['LotArea', 'HouseAge', 'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','NoBedroom']].copy()
Xb=numericategory.values
sc_Xb=StandardScaler()
Xb = sc_Xb.fit_transform(Xb)


#Step 5 : Handling Outliers 
sns.boxplot(x=data['GrLivArea'])
oplot(data['GrLivArea'])
#Deleting outliers
data = data.drop(data[(data['GrLivArea']>4000) & (data['SalePrice']<300000)].index)
oplot(data['GrLivArea'])

#Target variable analysis
distributionplot(data['SalePrice'])
#Skewness is asymmetry in a statistical distribution, in which the curve appears distorted or skewed either to the left or to the right
#we use log for target variable to make more normal distribution , Outliers handled using transformation
data['SalePrice'] = np.log1p(data['SalePrice'])
distributionplot(data['SalePrice'])





