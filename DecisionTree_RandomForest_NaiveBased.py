# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:43:18 2020

@author: Manish
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Manish\Desktop\ML\knn_data.csv')
X = dataset.iloc[:, [1,2, 3]].values
y = dataset.iloc[:, 4].values





from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder=LabelEncoder()
X[:,0]=labelencoder.fit_transform(X[:,0])
X
onehotencoder=OneHotEncoder(categorical_features=[0]) #3 is index of attribute
X=onehotencoder.fit_transform(X).toarray()
X

#remove 1st column as it is not making any contribution 00,01,10 are denoting 3 categories
X=X[:,1:]




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,1:] = sc.fit_transform(X_train[:,1:])
X_test[:,1:] = sc.transform(X_test[:,1:])

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)


from sklearn.naive_bayes import GaussianNB
classifier2=GaussianNB()
classifier2.fit(X_train,y_train)


from sklearn.neighbors import KNeighborsClassifier
classifier3=KNeighborsClassifier()
classifier3.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
classifier4=RandomForestClassifier()
classifier4.fit(X_train,y_train)
classifier4.feature_importances_  # max value is more important


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X_train,y_train)




# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score

c_val=cross_val_score(classifier,X,y,cv=10,verbose=1)
Tree_accuracy=sum(c_val)/10
Tree_accuracy



c_val2=cross_val_score(classifier2,X,y,cv=10,verbose=1)
Naive_accuracy=sum(c_val2)/10
print(Naive_accuracy)


c_val3=cross_val_score(classifier3,X,y,cv=10,verbose=1)
KNN_accuracy=sum(c_val3)/10
print(KNN_accuracy)

c_val4=cross_val_score(classifier4,X,y,cv=10,verbose=1)
RFC_accuracy=sum(c_val4)/10
print(RFC_accuracy)



# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()