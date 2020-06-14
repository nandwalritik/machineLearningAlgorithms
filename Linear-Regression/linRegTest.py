import LinearRegression as LR
import numpy as np 

def normalize(X):
	return (X - np.mean(X,axis=0))/np.std(X,axis=0)

data = np.loadtxt('data.txt',dtype=float,delimiter=',')

size = data.shape[0]

X,Y = data[:,0:2],data[:,-1]
X = normalize(X)

testSize    = (int)(0.3*size)
trainSize   = size - testSize 

X_train,X_test,y_train,y_test = X[:trainSize],X[trainSize:],Y[:trainSize],Y[trainSize:]

train = LR.LinearRegression()
train.setter(X_train,y_train)

test  = LR.LinearRegression()
test.setter(X_test,y_test)

train.gradientDescent()
print("Accuracy of model   	"+str(train.accuracy(test)))
print("Coeficients		"+str(train.coef_()[0]))
print("intercept Term		"+str(train.intercept()))

# Scikit Learn Implementation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred	  =   regressor.predict(X_test)

error = 100*((y_pred-y_test)/y_test)
print("Using scikit Learn :-")
print("Accuracy of Model    "+str(100-np.abs(np.mean(error))))
print("Coeficients          "+str(regressor.coef_))
print("Intercept Term		"+str(regressor.intercept_))