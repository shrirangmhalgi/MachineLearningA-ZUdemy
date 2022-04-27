# random forest is a version of ensemble learning
# other versions are there such as gradient boosting
# ensemble learning means take the same algorithm or combine multiple algorithms to create a much powerful model than the orignal one
# step 1: pick at random K data points from the training set
# step 2: build a decision tree associated to these K data points
# step 3: choose the number Ntree of trees you want to build and repeat steps 1 and 2
# step 4: For a new data point, make each one of your Ntree trees predict the value of Y to for the data point in question, and assign the new data point the average across of the all predicted y values 
# improves the accuracy of the prediction as it predicts the values based on the forest of trees and not a single tree
# take a wild guess game which is there at the fair
# basically you learn what people have guessed and as the guesses are normally distributes, you will take the median/mean and guess your number and by doing so there are chances of you to predict the value which is the closest number in the jar