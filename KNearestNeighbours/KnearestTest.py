import KnearestNeighbours as KNN 
import numpy  as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
def normalize(X):
	return (X-np.mean(X))/(np.std(X))


iris 		= 	load_iris()

X           =   iris.data
y 			=	iris.target

X,y = shuffle(X,y,random_state=0)	

sz   		=   iris.data.shape[0]
tsz			=   (int)(0.3*sz)
trainsz		=   sz-tsz
X_train,X_test,y_train,y_test = X[:trainsz,:],X[trainsz:,:],y[:trainsz],y[trainsz:]

# print(y_test)
train = KNN.KNearestNeighbours()
train.setter(X_train,y_train,5)

test  = KNN.KNearestNeighbours()
test.setter(X_test,y_test)
print(train.accuracy(test))

plt.scatter(X_train[:,2],X_train[:,3],c=y_train[:])
plt.title("Train Set")
plt.show()

plt.scatter(X_test[:,2],X_test[:,3],c=train.predict(X_test))
plt.title("Test set(class based Implementation)")
plt.show()



# Scikit-Learn Implementation
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))

plt.scatter(X_test[:,2],X_test[:,3],c=y_pred[:])
plt.title('Test Set as per Scikit-Learn')
plt.show()