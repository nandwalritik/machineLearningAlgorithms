import numpy as np 

class LogisticRegression:
	def setter(self,Xin,Yin):
		self.X 			= 	Xin
		self.Y 			= 	Yin
		self.m 			= 	Yin.size
		self.coef_ 		= 	np.random.rand(Xin.shape[1])
		self.intercept 	=   np.random.rand(1)

	def sigmoid(self,x):
		g = 1/(1+np.exp(-x))
		return g

	def computeCost(self,Lambda=0):
		h_theta 	= 	self.sigmoid(np.dot(self.X,self.coef_)+self.intercept)
		# regularize	=	(Lambda/(2*self.m))*(np.sum(np.square(self.coef_)))
		J_theta     =   -(np.dot(self.Y.T,np.log(h_theta))+np.dot(1-self.Y.T,np.log(1-h_theta)))/(self.m) + regularize
		return J_theta

	def gradientDescent(self,Lambda=0,alpha=0.01,iterations=2000):
		for i in range(1,iterations):
			h 		   = self.sigmoid(np.dot(self.X,self.coef_)+self.intercept)
			# print(h.shape)
			# print(self.Y.shape)
			# print(self.X.shape)
			tempCoef_  		=  self.coef_ 		- (alpha/self.m)*(np.dot(self.X.T,h-self.Y))  +(Lambda/(self.m))*self.coef_
			tempIntercept	=  self.intercept   - (alpha/self.m)*(np.sum(h-self.Y))
			self.coef_   	= tempCoef_
			self.intercept 	= tempIntercept
			# print("Iteration    ",str(i),"   cost   ",str(self.computeCost(Lambda)))
			# print(self.coef_," ",self.intercept)
	def stochasticGradientDescent(self,batchSize,Lambda=0,alpha=0.01,iterations=2000,shuffle=True):
		if shuffle :
			np.random.shuffle([self.X,self.Y])
		for i in range(1,iterations):
			for j in range(0,self.m,batchSize):
				X_mini = self.X[j:j+batchSize,:]
				y_mini = self.Y[j:j+batchSize]
				sz = y_mini.size
				if(y_mini.size == 0):
					break
				h_theta 		= self.sigmoid(np.dot(X_mini,self.coef_)+self.intercept)
				tempCoef_ 		= self.coef_ - (alpha/sz)*(np.dot(X_mini.T,(h_theta-y_mini)))
				tempIntercept	= self.intercept  - (alpha/sz)*(np.sum(h_theta-y_mini))
				self.coef_ = tempCoef_
				self.intercept	= tempIntercept
	def predict(self,X):
		return self.sigmoid(np.dot(X,self.coef_)+self.intercept)>=0.5

	def accuracy(self,test):
		prediction = self.predict(test.X)
		Accuracy   = (sum(prediction==test.Y)/test.m)*100
		return Accuracy

	def coeficient(self):
		return [self.coef_]

	def intercept_(self):
		return [self.intercept]	
