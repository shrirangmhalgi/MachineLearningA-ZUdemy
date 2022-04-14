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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train)
print(x_test)
print(y_train) 
print(y_test)

# feature scaling
# some features are dominated hence to reduce domination we use feature scaling
# used for certain machine learning 
# 2 feature scaling techniques are 
# 1. standarization -> (x - mean(x)) / standard_deviation(x) -> put the values of all the features between -3 and +3
# 2. normalization -> (x - min(x)) / (max(x) - min(x)) -> put the values of all the features between 0 and +1
# what is prefered? Normalization is recommended when you have normal distribution in most of your features. Standardisation works all the time. 
# recommended to use standardisation as it works all the time
# apply the scaling on x_train and x_test
# do we have to apply feature scaling to the dummy variables (categorical/one hot encoded variables) in the dataset?
# NO. we need all the values of the features in the same range. standardization standardises values mean = 0 and stardard deviation is 1 unit variance (between -3 and +3) our dummmy variables are between the range, hence Standarization will make it worse as we will lose on the interpretation of the variables 
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
x_train[:, 3:] = standard_scaler.fit_transform(x_train[:, 3:])
# features should be scaled by the same scaler which was used upon the train data.. no new scaler class is needed
# the ml model will be trained using the scaler hence to make the scaler ad predictions congruent, we need to use the same scaler which was used to train the model
x_test[:, 3:] = standard_scaler.transform(x_test[:, 3:])
print(x_train)
print(x_test)