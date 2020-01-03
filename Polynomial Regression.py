# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 08:22:37 2020

@author: Manish
"""
#polynomial Regression 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('C:\\Users\Manish\Desktop\Ayush\Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
x
y

#fitting linear reg to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting poly reg to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
# or
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

#visualising the linear regression results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Label')
plt.ylabel('salary')
#plt.show()

#visualising the polynomial regression results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(x_poly),color='red')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Label')
plt.ylabel('salary')
plt.show()

# lin_reg_2.predict([[6.5]])



