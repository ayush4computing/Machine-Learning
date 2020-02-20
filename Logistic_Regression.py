# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:24:30 2020

@author: Manish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('C:\\Users\Manish\Desktop\ML\knn_data.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
x
y

dataset.isnull().any()

#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=0)


from sklearn.preprocessing import StandardScaler
# it zero the mean and one the variance to get uniform distributed curve
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_train
x_test=sc.transform(x_test)
x_test

#fitting K-NN to Training set 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x,y)



#Predicting the Test Set Results
y_pred=classifier.predict(x_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)



# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()












