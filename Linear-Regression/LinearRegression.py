import numpy as np
class LinearRegression:
	
	# Part 1 setting parameters
	def setter(self,Xin,Yin):
		self.X 				= 	Xin
		self.Y 				= 	Yin
		self.m 				= 	Yin.size
		self.coeficients_ 	= 	np.random.rand(Xin.shape[1])
		self.intercept_ 	=   np.random.rand(1)
	
	# Part 3 Gradient Descent and Cost Calculation
	def computeCost(self):
		h_theta = np.dot(self.X,self.coeficients_)+self.intercept_
		J 		= (np.sum(np.power((h_theta - self.Y),2)))/(2*self.m)
		return J	

	def gradientDescent(self,alpha=0.01,iterations=2000):
		for i in range(1,iterations):
			h 				 	= 	np.dot(self.X,self.coeficients_) + self.intercept_
			temp_coeficients 	= 	self.coeficients_ -  (alpha/(self.m))*(np.dot(self.X.T,(h-self.Y)))
			temp_intercept   	= 	self.intercept_   -  (alpha/(self.m))*(np.sum(h-self.Y))
			self.coeficients_ 	= 	temp_coeficients
			self.intercept_		= 	temp_intercept
			# print("Iteration   "+str(i)+"   Cost   "+str(self.computeCost()))	

	# Prediction & Accuracy
	def predict(self,x):
		return np.dot(x,self.coeficients_)+self.intercept_

	def accuracy(self,test):
		predicttion = self.predict(test.X)
		error       = ((predicttion-test.Y)/test.Y)*100 
		return (100-np.mean(error))

	def coef_(self):
		return [self.coeficients_]

	def intercept(self):
		return self.intercept_		
