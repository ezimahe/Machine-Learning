#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:45:11 2019

@author: eleanorezimah
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 01:26:48 2019

@author: Owner
"""

from sklearn.linear_model import Lasso

def lasso_regression1(data, predictors, alpha, models_to_plot={}):
    '''
    Function to apply Lasso regression on specific data with mentioned alpha value
    
    '''
    # Defining the Lasso Regression model with specific Alpha value
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    # Fitting the given data to the Lasso regression model 
    lassoreg.fit(data[predictors].T,data[1])
    # Predicting the outputs
    y_pred = lassoreg.predict(data[predictors].T)
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        # Plot the Converged Lesso regressor hypotenus line for the given data
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data[0],y_pred)
        plt.plot(data[0],data[1],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    # Calculating the residual sum of squares error for the specific fitted data
    rss = sum((y_pred-data[1])**2)
    ret = [rss]
    # Appending the Lasso Regression Fit model intercept to the list
    ret.extend([lassoreg.intercept_])
    # Appending the Lasso Regression Fit model coefficients to the list
    ret.extend(lassoreg.coef_)
    return ret

#Importing libraries. The same will be used throughout the article.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10

#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducability
y = np.sin(x) + np.random.normal(0,0.15,len(x))
#data = np.concatenate([x[:,None],y[:,None]],axis=1)
data = np.vstack([x,y])
plt.plot(data[0],data[1],'.')

# Defining polynomial features for the given X data by calculating power coeffiecients from 2 to 16
for i in range(2,16):  #power of 1 is already there
    #new var will be x_power
    # Appending Each polynomial Feature to the Given X data
    data = np.vstack([data,data[0]**i])
    
    
#Initialize predictors to all 15 powers of x
predictors=[0]
predictors.extend([i for i in range(2,16)])

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

#Define the models to plot
models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#Iterate over the 10 alpha values
coef_matrix_lasso = []
for i in range(10):
    # Calling Lasso regression Function for each different Alpha values
    coef_matrix_lasso.append(lasso_regression1(data, predictors, alpha_lasso[i], models_to_plot))


