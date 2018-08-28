# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:17:08 2018

@author: Asus SLR points x and y
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('puckData.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

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
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.title('Ball Location (Training set)')
plt.xlabel('X axis point')
plt.ylabel('Y axis point')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.title('Ball Location (Test set)')
plt.xlabel('X axis point')
plt.ylabel('Y axis point')
plt.show()

