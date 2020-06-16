import numpy as np
import matplotlib.pyplot as plt
class KMeansClustering:
	def setter(self,Xin):
		self.X = Xin
		self.m = Xin.shape[0]
		self.Centroids = np.empty(self.m,dtype=int)

	def initialize_centroids(self,K):
		centroids = self.X
		np.random.shuffle(centroids)
		return centroids[:K]

	def assign_closest_centroid(self):
		for i in range(self.m):
			self.Centroids[i] = np.argmin(np.sqrt(np.sum((self.X[i]-self.centroid)**2,axis=1)),axis=0)
	def update_centroid(self,K):
		for i in range(K):
			matchIndices = np.argwhere(i == (self.Centroids))
			self.centroid[i] = np.mean(self.X[matchIndices,:],axis=0)
		
	def clustering(self,K,iterations=100):
		self.centroid = self.initialize_centroids(K)
		for i in range(1,iterations):			
			self.assign_closest_centroid()
			self.update_centroid(K)
		# print(self.centroid)

	def predict(self,X):
		m = X.shape[0]
		clust = np.zeros(m,dtype=np.int64)
		for i in range(m):
			clust[i]=np.argmin(np.sqrt(np.sum((X[i] - self.centroid)**2,axis=1)),axis=0)	
		return clust	
	def k_clusters_(self):
		return self.centroid	