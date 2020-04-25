# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:54:39 2020

@author: Shrikant Agrawal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Always set your working directory by pressing F5

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Divide dataset into X and Y, train and test is not req because of less data
x= dataset.iloc[:,1:2].values
y=dataset.iloc[:,2]

# Fitting Random Forest Regression to dataset
 # n_estimator value means the number of decision trees we are going to use
 # It will create 10 features and then take a mean of those 10 outputs and predict the output
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

""" Random forest is a classifier for various decision trees, n_estimator specifies how many 
decision trees we are using"""

#Predicting the new results
y_pred = regressor.predict(6.5)

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.show()

# Visulizing Random Forest Regression results (Higher Resolution)
x_grid = np.arange(min(x),max(x), 0.01)
x_grid= x_grid.reshape( (len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Random Forest Regression')
plt.show()