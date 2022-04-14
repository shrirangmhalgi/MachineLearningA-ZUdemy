# simple linear regression -> y = b0 + b1*x1 
# y -> dependent variable
# x -> independent variable (direct or implied association which casues dependent variable to change)
# b1 -> (slope of the line) coefficient of independent variable it defines how a unit change in x1 how it affects unit change in y. The connector between y and x
# b0 -> constant where the line cuts the y axis
# regression draws a line that best fits your data uses ordinary least squares method
# how the best fitting line is found?
# it draws a perpendicular to the line and takes the sum of the difference of the squares and chooses the line with minimum sum
# sum of (actual value - model predicted value)squared
# linear regression draws lots of these lines
# this is the ordinary least square method 
# branch  of ML which predicts continuous numbers. eg salary, temperature or continuous numerical value
# regression is used when you have to predict a continuous real value
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  

# reading the dataset
salary_dataset = pd.read_csv("simple_linear_regression_salary_data.csv")

# splitting the dependent and independent variables
x = np.array(salary_dataset.iloc[:, :-1].values)
y = np.array(salary_dataset.iloc[:, -1].values)

# splitting the data into train test set  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# reshape the array as standard scaler takes 2d array
# x_train.reshape(1, -1)
# x_test.reshape(1, -1)

# applying feature scaling on dependent variable
# x_standard_scaler = StandardScaler()
# x_train[:, :] = x_standard_scaler.fit_transform(x_train[:, :]) 
# x_test[:, :] = x_standard_scaler.transform(x_test[:, :])

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# training the regression model
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)

# predicting the test result
y_pred = linear_regression.predict(x_test)

# visualization of training set result
# trained and predicted salary printing
# x -> number of years of experience 
# y -> salary of the employee
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, linear_regression.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualization of test set results
plt.scatter(x_test, y_test, color='red')
# the predicted salary will be on the same regression line hence we need to not need to change the x_train and linear_regression.predict(x_train) line
plt.plot(x_train, linear_regression.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
