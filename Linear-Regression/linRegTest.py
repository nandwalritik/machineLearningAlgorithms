import LinearRegression as LR
import numpy as np 
import pandas as pd 

data = np.loadtxt('data.txt',dtype=float,delimiter=',')
# print(data)
size = data.shape[0]
print(size)
X,Y = data[:,0:2],data[:,-1]
# print("This is X",X)
# print("This is Y",Y)
# taking 60 to 40 for train test
testSize 	= (int)(0.3*size)
trainSize 	= size - testSize	 
# print(testSize)
X_train,X_test,y_train,y_test = X[:trainSize],X[trainSize:],Y[:trainSize],Y[trainSize:]
# print("Xtrain ",X_train)
# print("ytrain ",y_train)
# print("Xtest ",X_test)
# print("yTest ",y_test)

train = LR.LinearRegression()
train.setter(X_train,y_train)

test  = LR.LinearRegression()
test.setter(X_test,y_test)

train.gradientDescent()
print("Accuracy of model   ",train.accuracy(test))
