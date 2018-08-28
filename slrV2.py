# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:17:08 2018

@author: Asus
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('XYAxisTake0.csv')
y = dataset.iloc[:, :1].values
l = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
y_train, y_test, l_train, l_test = train_test_split(y, l, test_size = 2/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(y_train, l_train)

# Predicting the Test set results
l_pred = clf.predict(y_test)

# Visualising the Training set results
plt.scatter(y_train, l_train, color = 'red')
plt.title('Ball Location (Training set)')
plt.xlabel('Y axis point')
plt.ylabel('left or right')
plt.show()

# Visualising the Test set results
plt.scatter(y_test, l_test, color = 'red')
plt.title('Ball Location (Test set)')
plt.xlabel('Y axis point')
plt.ylabel('left or right')
plt.show()