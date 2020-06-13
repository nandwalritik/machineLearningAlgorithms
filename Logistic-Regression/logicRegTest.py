import numpy as np 
import LogisticRegression as LR
data = np.loadtxt('data1.txt',dtype=float,delimiter=',')
# print(data.shape)
# print(data)
size = data.shape[0]
testSize = (int)(0.2*size)
trainSize = size - testSize
X = data[:,:2]
y = data[:,-1]
X_train,X_test,y_train,y_test = X[:trainSize],X[trainSize:],y[:trainSize],y[trainSize:]
# print("X_train")
# print(X_train)
# print("X_test")
# print(X_test)
# print("y_train")
# print(y_train)
# print("y_test")
# print(y_test)

train = LR.LogisticRegression()
train.setter(X_train,y_train)
train.gradientDescent()

test = LR.LogisticRegression()
test.setter(X_test,y_test)
print(train.accuracy(test))
