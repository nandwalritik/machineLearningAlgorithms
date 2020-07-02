import numpy as np
class DecisionTreeClassifier:
	def setter(self,Xin,yin,max_depth):
		self.max_depth 	= max_depth
		self.X 			= Xin
		self.y 			= yin
		self.features 	= Xin.shape[1] # num of features in data
		self.n_classes_ = len(np.unique(yin))
		self.m 			= yin.size
		self.giniC 		= []
		self.th 		= []


	def bestSplit(self,X,y):
		''' finding best split for a node'''
		
		m = y.size
		#print(y.shape)
		if len(set(y)) <= 1:
			return None,None

		# Count of each class in current Node
		# cl,num_parent = np.unique(y,return_counts=True)
		num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
		# print(cl,num_parent)
		best_gini = 1.0 - sum((n/m)**2 for n in num_parent)
		best_idx,best_thr = None,None

		# loop through all features
		for idx in range(self.features):
			# sorting the data along selected feature
			thresholds,classes = zip(*sorted(zip(X[:, idx], y)))
			num_left 	= [0]*self.n_classes_
			num_right 	= num_parent.copy()
			# num_right 	= [0]*self.n_classes_
			# print("Before loop "+str(num_left)+" "+str(num_right))
			# possible split positions
			for i in range(1,m):
				c = classes[i-1]
				num_left[c] 	+= 	1
				num_right[c]	-= 	1
				gini_left = 1.0-sum((num_left[x]/i)**2 for x in range(self.n_classes_))
				gini_right= 1.0-sum((num_right[x]/(m-i))**2 for x in range(self.n_classes_))

				# Gini impurity as per the weighted gini impurity of the children
				gini = (i*gini_left + (m-i)*gini_right)/m 
				# print("GiniComp " + str(gini)+" "+str(best_gini))
				# for making sure that two points with same threshold don't gets splitted
				if thresholds[i] == thresholds[i-1]:
					continue
				if gini < best_gini:
					best_gini = gini
					best_idx  = idx
					best_thr  = (thresholds[i]+thresholds[i-1])/2		
			# print(num_left,num_right)
		# print(best_idx,best_thr)
		self.th.append([best_idx,best_thr])
		self.giniC.append([best_idx,best_gini])
		return best_idx,best_thr
	
	def fit(self):
		'''Building a decision tree classifier'''
		self.tree = self._growTree(self.X,self.y)

	def _gini(self,y):
		'''
			function for computing gini impurity
		'''	 
		m 		= y.size
		clss 	= np.unique(y)
		gini 	= 1  
		sub		= 0
		for x in clss:
			prob = (np.sum(y==x))/m
			prob = prob**2
			sub+=prob
		gini -= prob 	
		return gini

	def _growTree(self,X,y,depth=0):
		'''Building a decision Tree by recurssively finding the best split'''
		cl,num_samples_per_class = np.unique(y,return_counts=True)
		#print(cl,num_samples_per_class)
		# #print(num_samples_per_class)
		predicted_class = cl[np.argmax(num_samples_per_class)]
		#print(predicted_class)
		#print(self._gini(y))
		node = Node(
				gini 					= 	self._gini(y),
				num_samples 			= 	y.size,
				num_samples_per_class 	= 	num_samples_per_class,
				predicted_class 		= 	predicted_class
			) 		
		# split recurssively until maximum depth is reached
		if depth < self.max_depth:
			idx,thr = self.bestSplit(X,y)
			if idx is not None:
				indices_left 		= 	X[:,idx] < thr
				# print("index "+str(indices_left))
				X_left,y_left 		= 	X[indices_left],y[indices_left] 
				X_right,y_right 	= 	X[~indices_left],y[~indices_left]
				node.feature_index 	= 	idx
				node.threshold 		= 	thr 
				node.left 			= 	self._growTree(X_left,y_left,depth+1)
				node.right			= 	self._growTree(X_right,y_right,depth+1)

		return node		

	def predict(self,X):
		predictions = []
		node = self.tree
		for x in X:
			while node:
				if(node.left == None and node.right == None):
					break
				if x[node.feature_index] < node.threshold and node.left != None:
					node = node.left
				else:
					if node.right != None:
						node = node.right
			prediction = node.predicted_class
			predictions.append(prediction)
		return predictions

	def accuracy(self,X,y):
		predictions = self.predict(X)
		print(predictions)
		print(self.th)
		print(self.giniC)
		return 100*(np.sum(predictions == y)/len(y))

# class for implementing Node
class Node:
	def __init__(self,gini,num_samples,num_samples_per_class,predicted_class):
		self.gini 					=		gini
		self.num_samples			=		num_samples
		self.num_samples_per_class	=		num_samples_per_class
		self.predicted_class		=		predicted_class
		self.feature_index			=		0
		self.threshold 				=		0
		self.left 					= 		None
		self.right					= 		None			
					

		