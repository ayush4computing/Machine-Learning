# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 09:21:53 2020

@author: Manish
"""

#multiple linear regression
#p-value should be lowest. lower the p-value , more the significance of attribute

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('C:\\Users\Manish\Desktop\Ayush\\50_Startups.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:,4].values
x
y

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])
x
onehotencoder=OneHotEncoder(categorical_features=[3]) #3 is index of attribute
x=onehotencoder.fit_transform(x).toarray()
x

#remove 1st column as it is not making any contribution 00,01,10 are denoting 3 categories
x=x[:,1:]

#splitting dataset into training and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
i1=mean_squared_error(y_test,y_pred)



# Building the optimal model using backward elimination
import statsmodels.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1) #appending 1, 50 rows  1 column of 1 after it values of x 

x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit() 
regressor_OLS.summary()


x_opt=x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


#own
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_opt,y,test_size=.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn.metrics import mean_squared_error
i1
i2=mean_squared_error(y_test,y_pred)
i1-i2





