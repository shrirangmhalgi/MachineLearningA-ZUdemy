# R squared method
# simple linear regression
# sum (yi - yi^)^2 -> min
# basically we are taking the minimum of sum of the squares of diffrences of actual point and predicted points 
# SSres -> sum of squares of residuals = sum (yi - yi^)^2
# SStot -> total sum of squares of residuals = sum (yi - yavg)^2
# R^2 = 1 - (sum of squares of residuals / total sum of squares of residuals ) 
# in regression we are minimizing the sum of squares of residuals
# R^2 tells us how good is the line as compared to the avg line of your residuals
# if ideally SStot = 0 then R^2 = 1 which is not a case in real life scenario
# hence the closer the R^2 value is to 1 the better it is 
# can R^2 be negative?
# yes. SSres fits your data than the average line (it is hard)

# Adjusted R^2
# same thing applies to multiple linear regression
# SSres -> minimization is done
# R^2 is the goodness of fit (the greater it it the better)
# Problem: it starts to break when we add more and more variables
# once we add a new variable to model, it affects the model, the fact we are trying to minimize the sum of residuals is that either the new variable will help to minimize the sum, or it stays same
# Adj R^2 = 1 - (1 - R^2) * [(n - 1) / (n - p - 1)]
# p = number of regressors
# n = sample size
# Adj R^2 method penalizes you if you add independent variables that dont affect the model  