# y = b0 + b1x1 + b2x2 + .... + bnxn
# caveat associated with linear regression
# 1. linearity -> x and y have linear relation 
# 2. homoscedaticity -> varience of the residual is the same for any value of x
# 3. multivariate normality -> for any fixed value of x, y is normally distributed
# 4. Independence of errors -> the observations are independent of each other
# 5. lack of multicollinearity -> 
# check if assumptions are true.. then only you can be sure that you are building a good regression model
# profit = b0 + b1 * RnD Spend + b2 * Admin + b3 * State (not allowed as state is categorical variable)
# when you face categorical variables in regression then you need to create dummy variables  
# find the diffrent categories, and increase the number of columns
# just put 1 0 for the different categories 
# just include one of the dummy variables in the model not all (basically you will train the model for specific category) 
# dummy variable works as a switch
# model may seem biased but the coefficient is included in the b0 variable
# the coefficient of the dummy variable is actually calculated by taking the difference of the actual value and b0
# why it is bad to add both the dummy variables in the model? (dummy variable trap)
# D2 = 1 - D1 Always omit one dummy variables while building the model. 
# If there are 2 sets of dummy variables then apply the rule for each set
# what is P value?
# statistical significance is very important (intuition behind it)
# hypothesis testing -> we basically consider 2 cases. 
# 1. we get correct output (null hypothesis)
# 2. we dont get correct output (alternative hypothesis)
# P value is the probability of event happening where the null hypothesis is true
# we need to define a alpha value (confidence score) where the null hypothesis fails and we need to consider the alternative hypotheis
# how to build model step by step
# 1. we need to decide which ones to keep and which ones to throw out
# why to get rid of unwanted data? why cannot we use all the variables to train our model?
# a. garbage in garbage out. if you throw lot of stuff in the model then you will get garbage results
# b. at the end of the day you have to explain these varibales and understand the math behind it and also actually what it means and how it predicts the behavior of dependent variable to your manager or boss or mentor 
# keep only the important variables which actually are useful.
# 5 methods of building models
# a. All in
# b. Backward elimination
# c. Forward selection
# d. Bidirectional elimination
# e. Score comparision
# stepwise regression means b, c, and d sometimes stepwise regression means d. because d is the most general approach which is used
# a. All in : Just throw in all the variables. When we do this?
# 1. when you have prior knowledge that all the data is going to make sense and we have the domain knowledge or someone gave you the variables and said please build a model
# 2. there is a framework and you have to use all the variables to build the ML model
# 3. use this method when you are preparing for backward elimination
# b. Backward Elimination : 
#      step 1: select a significance level to stay in the model (eg. SL = 0.05 or 5%)
#      step 2: fit the full model with all the possible predictors  
# ---> step 3: consider the predictor with highest P value. If P > SL, go to step 4 else go to finish [model is ready]
# |    step 4: remove the predictor
# <--- step 5: fit the model without this variable after this go to step 3 again (once you remove the variable it affects everything and hence you need to refit the model)
# 
# c. Forward Selection: 
# step 1: select a significance level to enter the model (eg. SL = 0.05 or 5%)
# step 2: fit all the possible simple regression models y ~ xn. Select the one with lowest P value
# step 3: keep this variable and fit all the possible models with one extra predictor added to the one(s) you already have
# step 4: consider the predictor with lowest P value. If P < SL, go to step 3 else go to finish (keep the previous model while finishing)
# 
# d. Bidirectional Elimination:
# step 1: select a significance level to enter and stay in the model (eg. SLENTER = 0.05 and SLSTAY =  0.05) 
# --> step 2: perform the next step of forward selection (new variables must have P < SLENTER to enter)
# <-- step 3: perform all the steps of backward elimination (old variables must have P < SLSTAY to stay)
# step 4: no new variables can enter and no old variables can exit
# if step 4 satisfied then model is ready
# 
# All possible models: 
# step 1: select a criterion of goodness (eg akaike criterion)
# step 2: construct all the possible regression models 2n - 1 total combinations
# step 3: select the one with best criterion and your model is ready