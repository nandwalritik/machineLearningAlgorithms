import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
class NeuralLearn:
	def setter(self,Xin,Yin,layer_dims):
		self.X 			= Xin
		self.Y 			= Yin
		self.layer_dims = layer_dims
		self.m          = Yin.size
		self.parameters = {}
		self.grads		= {}
		self.L 			= len(self.layer_dims)
		self.initializeParameters()
		self.cachesPara = []
		self.cachesGrad = []
		self.loss       = []
		self.yhat 		= None
		
	def sigmoid(self,z):
		return 1/(1+np.exp(-z))
	
	def derSigmoid(self,z):
		x=self.sigmoid(z)
		return x*(1-x)
	
	def relu(self,z):
		return np.maximum(0,z)
	
	def derRelu(self,x):
		x[x<=0] = 0
		x[x>0]  = 1
		return x
	
	def initializeParameters(self):
		for l in range(1,self.L):
			self.parameters["W"+str(l)] = np.random.rand(self.layer_dims[l-1],self.layer_dims[l])*0.1
			self.parameters["b"+str(l)] = np.zeros(self.layer_dims[l],)
	
	def forwardProp(self):
		A_prev = self.X
		for i in range(1,self.L-1):
			# print(np.shape(A_prev),np.shape(self.parameters["W"+str(i)]))
			Z 		= np.dot(A_prev,self.parameters["W"+str(i)])+self.parameters["b"+str(i)]
			A 		= self.relu(Z)
			self.cachesPara.append([A,self.parameters["W"+str(i)],self.parameters["b"+str(i)],Z])
			A_prev	= A	
		Z = np.dot(A_prev,self.parameters["W"+str(self.L-1)])+self.parameters["b"+str(self.L-1)]
		A = self.sigmoid(Z)
		self.cachesPara.append([A,self.parameters["W"+str(i)],self.parameters["b"+str(i)],Z])	

	def cost(self):
		self.yhat 	= (self.cachesPara[-1][0])
		J 	= -(1/self.m)*(np.sum((self.Y*np.log(self.yhat)) + (1-self.Y)*np.log(1-self.yhat)))
		J	= np.squeeze(J)
		return J

	def softMax(self,x):
		expX = np.exp(x)
		return expX/expX.sum(axis=1,keepdims=True)

	def backwardProp(self):
		yhat  = self.cachesPara[-1][0]
		dl_wrt_yhat = -(np.divide(self.Y,yhat) - np.divide(1-self.Y,1-yhat))
		dl_wrt_sig	= yhat*(1-yhat)
		dA_prev = dl_wrt_yhat

		for l in reversed(range(len(self.cachesPara))):
			# print(l)
			if(l+1 == self.L-1):
				dZ = dA_prev*self.derSigmoid(self.cachesPara[l][-1])
			else:
				dZ = dA_prev*self.derRelu(self.cachesPara[l][-1])
			self.grads["dW"+str(l+1)] = np.dot(self.cachesPara[l-1][0].T,dZ) 
			self.grads["db"+str(l+1)] = np.sum(dZ,axis=0)
			if(l > 0):
				dA_prev = np.dot(dZ,(self.parameters["W"+str(l+1)]).T)	

	
	def gradientDescent(self,learningRate):	
		for i in range(len(self.cachesPara)):
			self.parameters["W"+str(i+1)] = self.parameters["W"+str(i+1)] - learningRate*self.grads["dW"+str(i+1)]
			self.parameters["b"+str(i+1)] = self.parameters["b"+str(i+1)] - learningRate*self.grads["db"+str(i+1)]

	def fit(self,iterations=300,learningRate=0.001):
		for i in range(iterations):
			self.forwardProp()
			self.backwardProp()
			self.gradientDescent(learningRate)
			self.loss.append(self.cost())
			# print("Iteration " + str(i+1)+" "+str(self.cost()))
			self.cachesPara.clear()
		
	def plot_loss(self):
		plt.plot(self.loss)
		plt.xlabel("Iteration")
		plt.ylabel("log loss")
		plt.title("Loss curve for training Using Class based Implementation")
		plt.show()

	def predict(self,X):
		z = np.dot(X,self.parameters["W1"])+self.parameters["b1"]
		a = self.sigmoid(z)
		for i in range(1,self.L-1):
			z = np.dot(a,self.parameters["W"+str(i+1)])+self.parameters["b"+str(i+1)]
			a = self.sigmoid(z)	
		return np.round(a)

	def accuracy(self,X,y):
		prediction = self.predict(X)
		return 100*(np.sum(y == prediction)/y.size)

			
