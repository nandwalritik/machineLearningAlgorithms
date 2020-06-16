import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import KMeansClustering as KMC


data  =  pd.read_csv('clustering.csv')
X = data.loc[:,["LoanAmount","ApplicantIncome"]].values

train = KMC.KMeansClustering()
train.setter(X)
train.clustering(3)
prediction = train.predict(X)

plt.scatter(X[:,0],X[:,1],c=prediction)
plt.show()

# ScikitLearn Implementation
from sklearn.cluster import KMeans 
model = KMeans(n_clusters=3,random_state = 0)
model.fit(X)
predicted = model.predict(X)
print(model.cluster_centers_)
plt.scatter(X[:,0],X[:,1],c=predicted)
plt.show()

