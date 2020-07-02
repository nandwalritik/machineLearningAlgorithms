import DecisionTree as Dtc 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle

data 	= 	np.loadtxt('breastCancer.txt',dtype=float,delimiter=',')
sz 		=	data.shape[0]
trSz 	= 	(int)(0.6*sz)

X 		= 	data[:,:-1]
y 		=	data[:,-1]
X,y 	= 	shuffle(X,y)

X_train,X_test,y_train,y_test = X[:trSz],X[trSz:],y[:trSz],y[trSz:]

train = Dtc.DecisionTreeClassifier()
train.setter(X_train,y_train)
train.predict(X_test)

print("Accuracy Using Class Based Implementation "+str(train.accuracy(X_test,y_test)))

# Scikit-Learn Implementation
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
predictiton = clf.predict(X_test)
print("Sklearn accuracy "+str(100*np.sum(predictiton == y_test)/len(y_test)))
