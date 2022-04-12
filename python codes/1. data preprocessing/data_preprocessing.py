# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:40:30 2022

@author: Shrirang Mhalgi
"""

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