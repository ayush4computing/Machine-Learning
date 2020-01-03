# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 09:35:43 2019

@author: student
"""
#Linear Regression
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('C:\\Users\Manish\Desktop\Ayush\Salary_Data.csv')
X=dataset.iloc[:,0:1].values
X
y=dataset.iloc[:,1].values
y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)
x_train
x_test


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results
y_pred=regressor.predict(x_test)

#visualising the trainig set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#visualising the trainig set results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(x_test,y_test,color='blue')
plt.scatter(x_test,y_pred,color='red')