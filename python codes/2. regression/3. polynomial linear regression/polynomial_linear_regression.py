# y = b0 + b1x1 + b2x1^2 + .... + bnx1^n
# limited use cases are for polynomial linear regression
# eg. disease spread
# even after there are different powers for x, why is it called polynomial 'linear' regression? 
# when we are talking about polynomial linear regression, we are not talking about x but we are talking about the constant b0, b1, etc
# y is a function of x can this function be expressed as a linear combination of the coefficients ultimately, they are the unknowns
# our goal is to find the values of the coefficients and use those coefficients to plug in x and ultimately find the value of y

# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reading the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# training linear regression model on whole dataset
from sklearn.linear_model import LinearRegression
linear_regression_regressor = LinearRegression()
linear_regression_regressor.fit(x, y)

# training polynomial regression model on whole dataset
# 2 step process
# 1. create the matrix of features x, x^2, ... x^n we have to find the optimal value of n
# 2. integrate the matrix of features into the linear regression model
from sklearn.preprocessing import PolynomialFeatures
polynomial_features = PolynomialFeatures(degree=5)
x_poly = polynomial_features.fit_transform(x)
polynomial_regression_regressor = LinearRegression()
polynomial_regression_regressor.fit(x_poly, y)

# visualization of linear regression results
plt.scatter(x, y, color='red')
plt.plot(x, linear_regression_regressor.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Grade')
plt.ylabel('Salary')
plt.show()

# visualization of polynomial regression results
plt.scatter(x, y, color='red')
plt.plot(x, polynomial_regression_regressor.predict(x_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Grade')
plt.ylabel('Salary')
plt.show()

# visualization of polynomial regression results for higher resolution and smoother curve
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, polynomial_regression_regressor.predict(polynomial_features.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Grade')
plt.ylabel('Salary')
plt.show()

# predicting a new result with linear regression
y_pred_linear = linear_regression_regressor.predict([[6.5]]) 
print(y_pred_linear)

# predicting a new result with polynomial regression
y_pred_poly = polynomial_regression_regressor.predict(polynomial_features.fit_transform([[6.5]]))
print(y_pred_poly)