# HyperParameters alpha = 0.001,iterations=300,multiplier=0.1,np.random.seed(2)
import NN as nClassifier
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore") #suppress warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# add header names
headers =  ['age', 'sex','chest_pain','resting_blood_pressure',  
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
        'num_of_major_vessels','thal', 'heart_disease']

heart_df = pd.read_csv('heart.dat', sep=' ', names=headers)


#convert imput to numpy arrays
X = heart_df.drop(columns=['heart_disease'])

#replace target class with 0 and 1 
#1 means "have heart disease" and 0 means "do not have heart disease"
heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)

y_label = heart_df['heart_disease'].values.reshape(X.shape[0], 1)

#split data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.3, random_state=2)

#standardize the dataset
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

print(f"Shape of train set is {Xtrain.shape}")
print(f"Shape of test set is {Xtest.shape}")
print(f"Shape of train label is {ytrain.shape}")
print(f"Shape of test labels is {ytest.shape}")

train = nClassifier.NeuralLearn()
train.setter(Xtrain,ytrain,[Xtrain.shape[1],8,1])
train.fit()
train.plot_loss()

print("Accuracy using class based implementation " + str(train.accuracy(Xtest,ytest)))

# # Scikit-Learn Implementation
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(8))
clf.fit(Xtrain,ytrain)
prediction = clf.predict(Xtest)
pre = [[x] for x in prediction]
pre = np.array(pre)
acc = 100*(np.sum(pre==ytest)/ytest.size)
print("Accuracy of Scikit-Learn Model "+str(acc))