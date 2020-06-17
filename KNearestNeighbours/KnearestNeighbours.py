import numpy as np
from collections import Counter
class KNearestNeighbours:
	def setter(self,Xin,Yin,kin=5):
		self.X 	=  Xin
		self.Y 	=  Yin
		self.m 	=  Yin.size
		self.k 	=  kin

	
	def euclideanDist(self,X1,X2):
		return np.sum(np.power((X1-X2),2),axis=1)

	def k_neighbours(self,Xin):
		arr = np.array(self.euclideanDist(self.X,Xin))
		ind = np.argsort(arr)
		# print(self.Y[ind[:self.k]])
		return self.Y[ind[:self.k]]

	def mean(self,labels):
		return sum(labels)/(len(labels))

	def mode(self,labels):
		# print(Counter(labels).most_common(1)[0][0])
		return Counter(labels).most_common(1)[0][0]

	def predict(self,Xin):
		sz 		= 	Xin.shape[0]
		y_ret 	= 	np.zeros(sz)
		for i in range(sz):
			y_ret[i] = self.mode(self.k_neighbours(Xin[i,:]))
		return y_ret
			
	def accuracy(self,test):
		prediction = self.predict(test.X)
		# print(prediction)
		# print(test.Y)
		accrcy 	   = (sum(prediction == test.Y)/test.m)*100
		return accrcy
			