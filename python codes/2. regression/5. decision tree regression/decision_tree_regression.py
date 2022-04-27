# CART : Classification and Regression Trees
# there are 2 types of decision trees : classification trees and regression trees
# regression trees are a bit more complex than classification trees
# consider a scatter plot which represents our data there are 2 independent variables x1 and x2 and we want to predict the 3rd varibale which is a dependent variable y
# the algorithm will be splitted into segments. just go to on splitting your data.
# how the splits are conducted is decided by algorithm and it actually involves information entrophy
# by each splitting are we deriving new information from the data? if yes then split if no then stop
# each split is called as a leaf
# it just creates a binary tree of conditions
# the leaf contains the dependent variable 
# the average of the terminal leaves is taken and then this is the value which is assigned to that leaf 

# the decision tree regression model is not really well adapted to simple datasets (only with 1 feature)
# no feature scaling required for decision tree and random forest regression as these are not equations they just give result by splitting the data

# importing the libraries
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# training the decision tree regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor(random_state=0)
decision_tree_regressor.fit(x, y)

# predicting a new result
y_pred = decision_tree_regressor.predict([[6.5]])
print(y_pred)

# visualization of decision tree regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, decision_tree_regressor.predict(x_grid), color='blue')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Truth or Bluff (Decision Tree Regressor)")
plt.show()

# visualization of decision tree regression results (low resolution) [this doesnt make sense as the predicted salary was on integer]
plt.scatter(x, y, color='red')
plt.plot(x, decision_tree_regressor.predict(x), color='blue')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Truth or Bluff (Decision Tree Regressor)")
plt.show()
