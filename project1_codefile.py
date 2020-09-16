# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#Reading the data from your files

data = pd.read_csv('advertising.csv')
data.head()

#To visualise data

fig, axs = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter', x = 'TV', y = 'Sales', ax = axs[0], figsize=(14,7))
data.plot(kind='scatter', x = 'Radio', y = 'Sales', ax = axs[1])
data.plot(kind='scatter', x = 'Newspaper', y = 'Sales', ax = axs[2])


#Creating x & y for linear regression

feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales


#Importing Linear Regression Algorithm

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)


print(lr.intercept_)
print(lr.coef_)


#Create a Dataframe with minimum and max value of table

X_new = pd.DataFrame({'TV':[data.TV.min(), data.TV.max()]})
X_new.head()


preds = lr.predict(X_new)
preds


data.plot(kind='scatter', x = 'TV', y = 'Sales')

plt.plot(X_new,preds,c='red',linewidth = 3)


import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV',data = data).fit()
lm.conf_int()


#Finding the Probablity Values
lm.pvalues


#Finding the R-Squared Values

lm.rsquared


#Multi Linear Regression
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales


lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)


lm = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()


