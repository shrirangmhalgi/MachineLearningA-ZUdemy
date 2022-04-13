# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:40:30 2022

@author: Shrirang Mhalgi
"""
#upper bound of the range is excluded in python

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the datasets
dataset = pd.read_csv('Data.csv')
# x -> matrix of features 
# y -> dependent variable vector
# iloc allows you to play with rows and columns of a dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

#taking care of missing data
#generally we do not keep any missing data as it can cause errors while training the model
#1. ignore the data (if you have a large dataset) the missing data constitutes only 1%
#2. replace the missing value by the average of the values of the column values
from sklearn.impute import SimpleImputer
simple_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
simple_imputer.fit(x[:, 1:3])
x[:, 1:3] = simple_imputer.transform(x[:, 1:3])
print(x)

#encoding categprical data
#encoding independent variable
#need to covert string into numbers Model can misinterpret if we give numbers. hence we use one hot encoding
#convert the categorical data into binary vector of 1's n 0's n increase the number of columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#remainder is used to keep the columns and transformer takes the encoder, what encoder, and index of the dataframe
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(column_transformer.fit_transform(x))
print(x)

#encoding dependent variable
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)

#splitting the data into train and test set
#do we have to apply feature scale before splitting the dataset into train and test set or after? 
# -> we apply the feature scaling. Feature scaling is to scale your features into a same scale, we do this because one feature should not dominate the another one which inturn should not be neglected by the machine learning model.
# We do feature scaling after splitting because the test set should be a brand new set using which you are going to evaluate the ML model.
# feature scaling gets mean and std deviation of the set, so if we apply the scaling before then the mean and std deviation will change. 
# Test set is the future set and if we apply scaling before then there will be information leakage for the test set.
# recommended size of split is 80 train 20 test
# random_state = 1 is fixing the seed to get same result (not required)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train) 
print(y_test)

# feature scaling
