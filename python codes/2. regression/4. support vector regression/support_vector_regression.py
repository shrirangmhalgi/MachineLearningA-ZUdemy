# invented by vladimir vapnik while working at bell labs
# svr, instead of line there is a tube of width epsilon (epsilon insensitive tube) which is measured perpendicular to x 
# the tube is basically margin of error where in the error inside the tube is disregarded
# the distance between the points and trendline is measured with regards to the epsilon tube and not the trend line. 
# the points outside the tube are called slack variables c* if the point is below the tube and c if point is above the tube 
# linear regression uses ordinary least square methods
# svr uses min (1/2 ||w||^2 + c * summation i = 1 to m (ci + ci*))
# why is it called support vector regression? because any point on the plot can be represented as a vector in 2d space and the points outside the tube basically act as a support and hence it is called support vector regression 
# non linear support vector regression (data is been trained on non linear svr)

# we apply feature scaling when the values do not get compensated while training the model as there is no explicit equation of dependent variables and there are no coefficients which compensate this thing
# linear regression doesnt take feature scaling as the coefficients are there to compensate the relation between dependent variable y and features x  

# importing the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

# applying feature scaling to dependent variables and independent variables so that the model doesnt ignore lesser values
# standard scaler class expects 2d array hence we need to reshape the array  
y = y.reshape(len(y), 1)
print(y)

# reason for creating 2 scaler objects is because when we do fit_tranform it calculates the mean and standard deviation of the data and as our data doesnt have same mean, hence we need 2 objects 
from sklearn.preprocessing import StandardScaler
standard_scaler_x = StandardScaler()
x = standard_scaler_x.fit_transform(x)
standard_scaler_y = StandardScaler()
y = standard_scaler_y.fit_transform(y)
print(x)
print(y)

# training the svr model on whole dataset
from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(x, y)

# predicting a new result
y_pred = standard_scaler_y.inverse_transform(svr_regressor.predict(standard_scaler_x.transform([[6.5]])))
print(y_pred)

# visualization of svr results
plt.scatter(standard_scaler_x.inverse_transform(x), standard_scaler_y.inverse_transform(y), color='red')
plt.plot(standard_scaler_x.inverse_transform(x), standard_scaler_y.inverse_transform(svr_regressor.predict(x)), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.show()

# visualization of svr result for higher resolution and smoother curve
x_grid = np.arange(min(standard_scaler_x.inverse_transform(x)), max(standard_scaler_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(standard_scaler_x.inverse_transform(x), standard_scaler_y.inverse_transform(y), color='red')
plt.plot(x_grid, standard_scaler_y.inverse_transform(svr_regressor.predict(standard_scaler_x.transform(x_grid))), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.show()
