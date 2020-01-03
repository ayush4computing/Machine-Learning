#preprocessor

import numpy as np
import pandas as pd
import matplotlib as plt

#importing the datasets
dataset=pd.read_csv('C:\\Users\\Manish\\Desktop\\Ayush\\Data.csv')
x1=dataset.iloc[:,:-1].values #-1 is excluded
x1
y1=dataset.iloc[:,4].values
y1


#Taking care of missing values

#axis=0 column 
#axis=1 rows

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x1[:,2:4]) #starts from 0 and 4 is excluded therefore we are reading only 2 & 3 column
x1[:,2:4]=imputer.transform(x1[:,2:4])
x1

#above can be performed with single line
# x1[:,2:4]=imputer.fit(x1[:,2:4]).transform(x1[:,2:4])


# Train Test Split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=.2,random_state=0)
# ctrl+i on func name to view its arguments

#the order above is fixed

x_train
x_test
y_train
y_test


#scaling says that we should have equal range in our dataset

from sklearn.preprocessing import StandardScaler
# it zero the mean and one the variance to get uniform distributed curve

sc_x=StandardScaler()
x_train[:,2:4]=sc_x.fit_transform(x_train[:,2:4])
x_train
x_test[:,2:4]=sc_x.transform(x_test[:,2:4])
x_test










