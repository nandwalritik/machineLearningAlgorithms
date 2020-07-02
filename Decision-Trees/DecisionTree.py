import numpy as np
class DecisionTreeClassifier:
	def setter(self,Xin,yin):
		self.X 			= Xin
		self.y 			= yin
		self.m 			= yin.size # number of training examples
		self.features 	= Xin.shape[1]
		self.Node 		= self.build_tree(self.X,self.y)

	def class_counts(self,y):
		"""Counts the number of each type of example in a dataset"""	
		counts = {} # a dictionary of label -> count
		for a in y:
			if a not in counts:
				counts[a] = 0
			counts[a] += 1
		
		return counts		

	def gini_cost(self,X,y):

		classCount 	= self.class_counts(y)
		_gini_Cost 	= 1
		sub			= 0
		for label in classCount:
			prob 	= 	classCount[label]/float(len(X))
			sub		+=	prob**2
		_gini_Cost -= sub
		
		return _gini_Cost

	def info_gain(self,right_X,right_y,left_X,left_y,current_uncertainity):
			frac = float(len(right_X))/(len(right_X)+len(left_X))
			return current_uncertainity - p*self.gini_cost(right_X,right_y)-(1-p)*self.gini_cost(left_X,left_y)		 


	class Question:
		'''A Question is used to partition a dataset'''
		def __init__(self,column,value):
			self.column = column
			self.value	= value

		def match(self,example):
			'''	This method is used to compare feature value in example to 
				feature value in this question.
			'''				
			value = example[self.column]
			if isinstance(value, int) or isinstance(value, float):
				return value >= self.value
			else:
				return value == self.value

	def partition(self,X,y,question):
		'''
			Partition a dataset
	
			For each row in the dataset,check if it is matching the question.
				if True:
					add it to trueRows
				else:
					add to falseRows		
		'''	
		right_X,right_y,left_X,left_y = [],[],[],[]
		for i in range(len(X)): # Traversal through len(X)
			if question.match(X[i]):
				right_X.append(X[i])
				right_y.append(y[i])
			else:
				left_X.append(X[i])
				left_y.append(y[i])
			return right_X,right_y,left_X,left_y
					
	def find_best_split(self,X,y):
		"""
			It finds the best threshold to be taken to have max Gini 
			of the split
		"""
		best_gain = 0
		best_ques  = None
		current_uncertainity = self.gini_cost(X,y)
		for feature in range(self.features):
			values = set([row[feature] for row in X])
			for val in values:
				question = self.Question(feature,val)
				right_X,right_y,left_X,left_y = self.partition(X,y,question)
				if(len(right_X) == 0 or len(left_X) == 0):
					continue
				gain = self.info_gain(right_X,right_y,left_X,left_y)	
				
				if gain > best_gain:
					best_gain,best_ques = gain,question

		return best_gain,best_ques				

	class Decision_Node:
		def __init__(self,question,true_branch,false_branch):
			self.question = question
			self.true_branch = true_branch
			self.false_branch = false_branch

	def build_tree(self,X,y):
		gain,question = self.find_best_split(X,y)
		if gain == 0:
			return self.class_counts(y)
		right_X,right_y,left_X,left_y = self.partition(X,y,question)
		right_branch	=	self.build_tree(right_X,right_y)
		left_branch		=	self.build_tree(left_X,left_y)
		return self.Decision_Node(question,right_branch,left_branch)

	def classify(self,Node,example):
		if isinstance(Node,dict):
			return Node
		else:
			if(Node.question.match(example)):
				return self.classify(Node.right_branch,example)
			else:
				return self.classify(Node.left_branch,example)
					

	def predict(self,X):
		predictions = []
		for x in X:
			d = self.classify(self.Node,x)
			val = list(d.values())
			key = list(d.keys())
			predictions.append(key[val.index(max(val))])
		return np.array(predictions)	
	def accuracy(self,X,y):
		predictions = self.predict(X)
		a = np.array(predictions==y)
		_accuracy = np.mean(a)*100
		return _accuracy


