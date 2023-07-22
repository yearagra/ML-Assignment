#!/usr/bin/env python
# coding: utf-8

# In[219]:


import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,precision_score,recall_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from statistics import mean
from math import *


# In[227]:


# DATASET

data = pd.read_csv('BitcoinHeistData.csv')
data = data.drop(['address'], axis=1)
data


# In[233]:


# ENCODING 

labelencoder_Y = LabelEncoder()
data['lable'] = labelencoder_Y.fit_transform(data['label'])
data


# In[261]:


data = data.sample(frac=1)
data


# In[185]:


# EDA REPORT 

profile = ProfileReport(data, title="Data Analysis Report")
profile.to_notebook_iframe()


# In[262]:


# TRAIN TEST VALIDATION SPLIT 

X= data.drop(columns='label')
Y=data[['label']]
def train_val_test_split(data, labels, train,val,test):
    print("Length: "+str(len(data)))
    
    train_data=data[:len(data)*0.7,:]
    train_labels=labels[:len(data)*0.7,:]
    
    val_data=data[len(data)*0.7:len(data)*0.85,:]
    val_labels=labels[len(data)*0.7:len(data)*0.85,:]
    
    test_data=data[len(data)*0.85:,:]
    test_labels=labels[len(data)*0.85:,:]
    
    return train_data,train_labels,val_data,val_labels,test_data,test_labels


# In[263]:


# SPLITTED DATA
print(X_train.shape, X_val.shape, X_test.shape)
print(Y_train.shape, Y_val.shape, Y_test.shape)


# In[265]:


# GINI INDEX

depth=[4,8,10,15,20]
trainingAcc=[]
testingAcc=[]

for i in depth:
    gini_tree = DecisionTreeClassifier(criterion='gini',max_depth=i)
    gini_tree = gini_tree.fit(X_train,Y_train)
    trainAcc = gini_tree.score(X_train,Y_train)
    testAcc = gini_tree.score(X_test,Y_test)
    print("Accuracy For Depth: ",i)
    print("Accuracy: ", trainAcc)
   # print("Accuracy test: ", testAcc)
    print()
    trainingAcc.append(trainAcc*100)
    testingAcc.append(testAcc*100)


# In[255]:


# ENTROPY

depth=[4,8,10,15,20]
trainingAcc=[]
testingAcc=[]

for i in depth:
    entropy_tree = DecisionTreeClassifier(criterion='entropy',max_depth=i)
    entropy_tree.fit(X_train,Y_train)
    trainAcc = entropy_tree.score(X_train,Y_train)
    testAcc = entropy_tree.score(X_test,Y_test)
    print("Accuracy For Depth: ",i)
    print("Accuracy: ", trainAcc)
    print()
    trainingAcc.append(trainAcc*100)
    testingAcc.append(testAcc*100)


# In[ ]:


# GETTING 50% DATASET RANDOMLY AND MAKING 100 DECISION TREES WITH (MAX DEAPTH 3)

def max_vote(predictions):
    final_prediction=[]
    for j in range(len(predictions[0])):
        maxi={}
        for i in predictions:
            if(i[j] in maxi):
                maxi[i[j]]+=1
            else:
                maxi[i[j]]=1
        max_key=max(maxi, key= lambda X: maxi[X])
        final_prediction.append(max_key)
    
    return final_prediction

stumps=[]
predictions=[]
for i in range(100):
    stumps.append(DecisionTreeClassifier(criterion="entropy",max_depth=3))
    X_train_frac=X_train.sample(frac=0.5)
    Y_train_frac=Y_train.loc[X_train_frac.index]
    stumps[i].fit(X_train,Y_train)
    predicts=stumps[i].predict(X_test)
    predictions.append(predicts)

final_prediction=max_vote(predictions)
accuracy=np.sum(np.array(final_prediction)==np.array(Y_test.to_list()))/len(Y_test)

print("Accuracy: "+str(accuracy))


# In[259]:


# ADABOOST

estimators = [4, 8, 10, 15, 20]
testAcc=[]
trainAcc=[]
valAcc=[]

for i in estimators:
    adaboost_tree = AdaBoostClassifier(n_estimators=i, base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=i))
    adaboost_tree.fit(X_train,Y_train)
    testAcc.append(adaboost_tree.score(X_test,Y_test))
    trainAcc.append(adaboost_tree.score(X_train,Y_train))
    valAcc.append(adaboost_tree.score(X_val,Y_val))
    print("Accuracy For Estimate: ",i)
    print("Training Accuracy: ", trainAcc[-1])
    print("Validating Accuracy: ", valAcc[-1])
    print("Testing Accuracy: ", testAcc[-1])
    

