# -*- coding: utf-8 -*-
"""Untitled10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MFznjwZE09yYd2CMA793Km9ymo0dCanZ

# Packages and Dataset
"""

import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

import io

from google.colab import drive
drive.mount('/drive')

#data file read
data = pd.read_csv('/drive/My Drive/MLAssignment4/population.csv') 
data

columns1=["AAGE", "ACLSWKR", "ADTIND", "ADTOCC",	"AHGA",	"AHRSPAY",	"AHSCOL",	"AMARITL",	"AMJIND",	"AMJOCC"	,"ARACE"	,"AREORGN",	"ASEX",	"AUNMEM"	,"AUNTYPE"	,"AWKSTAT",	"CAPGAIN",	"CAPLOSS",	"DIVVAL",	"FILESTAT"	,"GRINREG"	,"GRINST",	"HHDFMX",	"HHDREL",	"MIGMTR1",	"MIGMTR3",	"MIGMTR4",	"MIGSAME",	"MIGSUN",	"NOEMP"	,"PARENT",	"PEFNTVTY"	,"PEMNTVTY",	"PENATVTY"	,"PRCITSHP",	"SEOTR",	"VETQVA",	"VETYN"	,"WKSWORK",	"YEAR"]

columns2=[ "ACLSWKR",	"AHGA",	"AHSCOL",	"AMARITL",	"AMJIND",	"AMJOCC"	,"ARACE"	,"AREORGN",	"ASEX",	"AUNMEM"	,"AUNTYPE"	,"AWKSTAT",	"FILESTAT"	,"GRINREG"	,"GRINST",	"HHDFMX",	"HHDREL",	"MIGMTR1",	"MIGMTR3",	"MIGMTR4",	"MIGSAME",	"MIGSUN","PARENT",	"PEFNTVTY"	,"PEMNTVTY",	"PENATVTY"	,"PRCITSHP",	"VETQVA"	]

"""# Preprocessing"""

#Replace ? with NAN
for q in columns2:
  data[q] = data[q].mask(data[q].str.strip() == "?")
  data[q] = data[q].apply(lambda x: np.nan if x == '?' else x) 
  data[q].apply(lambda x: np.nan if str(x).find('?')>-1 else x)

data

#Null Values in Respective column
data.isnull().sum().sort_values(ascending = False).head(10)

#Checking the missing value and if they have more than 40% will remove them
perc = 40.0 
for c in columns2:
   print(c,data[c].isna().sum()/len(data))
min_count =  int(((100-perc)/100)*data.shape[0] + 1)
data = data.dropna( axis=1, 
                thresh=min_count)

"""# Imputation, Bucketization, One Hot Encoding"""

#Droping columns based on ratio with more than 85% 
az=[]
for col in data: 
  print(data[col].value_counts(ascending = True, normalize = True).max(), col)
for col in data:
  ratio = data[col].value_counts(ascending = True, normalize = True).max()
  if(ratio > 0.85): 
    az.append(col)

data=data.drop(az, axis = 1)

data.columns

"""Imputation"""

mode_append=[]
for column in data.columns:
    mode_append.append(data[column].mode()[0])
    data[column].fillna(data[column].mode()[0], inplace=True)
print(mode_append)

"""Bucketization"""

# Numercial Columns 
numercial_columns=["AAGE","WKSWORK"]

#Bucketizing
from sklearn.preprocessing import KBinsDiscretizer
def bucketize(data, c, b):
  Bucketizer=KBinsDiscretizer(n_bins=b, encode='ordinal', strategy='quantile')
  n1data=pd.cut(data[c], bins = b, labels = False).to_numpy()
  d1=Bucketizer.fit_transform(n1data.reshape(-1, 1))
  data[c] = d1

for c in numercial_columns:
  if(c=="AAGE"):
    bucketize(data,c,10)
  else:
    bucketize(data,c,5)

data

"""One-Hot Encoding"""

dt=data.copy()
for c in data.columns:

  dt= pd.get_dummies(dt, columns=[c], prefix = [c])

print(dt)

data1= dt.copy()
data2=dt.copy()

data2

dataframe = pd.concat([data, dt], axis=1, join='inner')

dataframe

"""# PCA"""

#PCA for dividng the dataset into samples and labels
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA().fit(data2)
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(0, 350, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 352, step=8)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.80, color='r', linestyle='-')
plt.text(0.5, 0.85, '80% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

a1=data2.copy()
pca = PCA(n_components = 32)
pca.fit(a1)
PCA_data = pca.transform(a1)

np.shape(PCA_data)
length_PCA_data= len(PCA_data)

"""# Clustering"""

from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.kmedians import kmedians
def KMEDIAN(reduced,value_k): 
  np.random.shuffle(reduced)
  k_median1=kmedians(reduced,np.copy(np.unique(reduced, axis=0)[:value_k]))
  k_median1.process()
  clus_distance=k_median1.get_total_wce()
  return clus_distance

# cluster distances store the avg within-cluster distance for k=[10,24]
cluster_distances=[]
idx=9
while(idx<24):
  y=PCA_data.copy()
  np.random.shuffle(y)
  k_median1=kmedians(y,np.copy(np.unique(y, axis=0)[:idx+1]))
  k_median1.process()
  clus_distance=k_median1.get_total_wce()
  cluster_distances.append(clus_distance)
  print(idx+1,clus_distance/(idx+1))
  idx=idx+1

clg=[]

for g in range(10,25):
    clg.append(cluster_distances[k-10]/g)

plt.plot([10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],clg,marker='*')
plt.grid()
plt.xlabel("No. of clusters")
plt.ylabel("AVERAGE DISTACNE ")
plt.title("K_MEDIAN CLUSTERING ")
plt.show()

#Converting Medians into np.array()
def get_total():
  cost = 0
  for i in range(len(md)):
    cluster = PCA_data[cluster[i]]
    cost_cluster = np.sum((cluster - md[i])**2, axis = 1, keepdims = True)
    cost += np.sum(cost_cluster)
  return cost/len(md)

# k median clustering on best value of k = 22
y=PCA_data.copy()
np.random.shuffle(y)
k_median1=kmedians(y,np.copy(np.unique(y, axis=0)[:22]))
k_median1.process()
clus_distance=k_median1.get_total_wce()
cluster = k_median1.get_clusters()
md = k_median1.get_medians()
print(22,clus_distance/22)

"""# Handling more than 50K Data

Dataset
"""

dataset = pd.read_csv("/drive/MyDrive/MLAssignment4/more_than_50k.csv")

columns1A=["AAGE", "ACLSWKR", "ADTIND", "ADTOCC",	"AHGA",	"AHRSPAY",	"AHSCOL",	"AMARITL",	"AMJIND",	"AMJOCC"	,"ARACE"	,"AREORGN",	"ASEX",	"AUNMEM"	,"AUNTYPE"	,"AWKSTAT",	"CAPGAIN",	"CAPLOSS",	"DIVVAL",	"FILESTAT"	,"GRINREG"	,"GRINST",	"HHDFMX",	"HHDREL",	"MIGMTR1",	"MIGMTR3",	"MIGMTR4",	"MIGSAME",	"MIGSUN",	"NOEMP"	,"PARENT",	"PEFNTVTY"	,"PEMNTVTY",	"PENATVTY"	,"PRCITSHP",	"SEOTR",	"VETQVA",	"VETYN"	,"WKSWORK",	"YEAR"]

columns2A=[ "ACLSWKR",	"AHGA",	"AHSCOL",	"AMARITL",	"AMJIND",	"AMJOCC"	,"ARACE"	,"AREORGN",	"ASEX",	"AUNMEM"	,"AUNTYPE"	,"AWKSTAT",	"FILESTAT"	,"GRINREG"	,"GRINST",	"HHDFMX",	"HHDREL",	"MIGMTR1",	"MIGMTR3",	"MIGMTR4",	"MIGSAME",	"MIGSUN","PARENT",	"PEFNTVTY"	,"PEMNTVTY",	"PENATVTY"	,"PRCITSHP",	"VETQVA"	]

dataset

for c in columns2A:
  print(c) 

  print(dataset[c])
  dataset[c] = dataset[c].mask(dataset[c].str.strip() == "?")
  dataset[c] = dataset[c].apply(lambda x: np.nan if x == '?' else x) 
  dataset[c].apply(lambda x: np.nan if str(x).find('?')>-1 else x)

dataset.isnull().sum().sort_values(ascending = False).head(10)

perc = 40.0
for c in columns2A:
   print(c,dataset[c].isna().sum()/len(dataset))
min_count =  int(((100-perc)/100)*dataset.shape[0] + 1)
dfA = dataset.dropna( axis=1, 
                thresh=min_count)

az=[]
for col in dataset: 
  print(dataset[col].value_counts(ascending = True, normalize = True).max(), col)
for col in dataset:
  ratio = dataset[col].value_counts(ascending = True, normalize = True).max()
  if(ratio > 0.85): 
    az.append(col)

dataset=dataset.drop(az, axis = 1)

for c in dataset.columns:
  print(dataset[c].isna().sum())

mode_append1=[]
for column in dataset.columns:
    print(column,dataset[column].mode()[0] )
    mode_append1.append(dataset[column].mode()[0])
    dataset[column].fillna(dataset[column].mode()[0], inplace=True)
print(mode_append1)

for c in dataset.columns:
  print(dataset[c].unique)

numercial_columns=["AAGE","CAPGAIN","DIVVAL","WKSWORK"]

#bucketizing
from sklearn.preprocessing import KBinsDiscretizer
def bucketize(dataset, c, b):
  Bucketizer=KBinsDiscretizer(n_bins=b, encode='ordinal', strategy='quantile')
  n1data=pd.cut(dataset[c], bins = b, labels = False).to_numpy()
  d1=Bucketizer.fit_transform(n1data.reshape(-1, 1))
  dataset[c] = d1

for c in numercial_columns:
  if(c=="AAGE" or c=="CAPGAIN" or c== "DIVVAL" or c=="WKSWORK"):
    bucketize(dataset,c,10)
  else:
    bucketize(dataset,c,5)

# one hot encoding
datasetA=dataset.copy()
for c in dataset.columns:

  datasetA= pd.get_dummies(datasetA, columns=[c], prefix = [c])

print(datasetA)

data1A= datasetA.copy()
data2A=datasetA.copy()

conclusionA = pd.concat([datasetA, datasetA], axis=1, join='inner')

# PCA ANALYSIS

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA().fit(datasetA)
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(0, 229, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 229, step=10)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.80, color='r', linestyle='-')
plt.text(0.5, 0.85, '80% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

a2=data2A.copy()
pca = PCA(n_components = 32)
pca.fit(data2A)
PCA_dataA = pca.transform(data2A)

from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.kmedians import kmedians
def KMEDIAN(reduced,value_k): 
  np.random.shuffle(reduced)
  k_median1=kmedians(reduced,np.copy(np.unique(reduced, axis=0)[:value_k]))
  k_median1.process()
  clus_distance=k_median1.get_total_wce()
  return clus_distance

cluster_distances1=[]
idx=9
while(idx<24):
  y=PCA_dataA.copy()
  np.random.shuffle(y)
  k_median1=kmedians(y,np.copy(np.unique(y, axis=0)[:idx+1]))
  k_median1.process()
  clus_distance=k_median1.get_total_wce()
  cluster_distances1.append(clus_distance/(idx+1))
  print(idx+1,clus_distance/(idx+1))
  idx=idx+1

zx1=[]
for k in cluster_distances1:
    zx1.append(k)

plt.plot([10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],zx1,marker='*')
plt.ylabel('Number of cluster')
plt.xlabel("average cluster distance")
plt.grid()
plt.legend()
plt.title("K median plot")
plt.show()

y=PCA_dataA.copy()
np.random.shuffle(y)
k_median1=kmedians(y,np.copy(np.unique(y, axis=0)[:22]))
k_median1.process()
clus_distance=k_median1.get_total_wce()
cluster1 = k_median1.get_clusters()
md1 = k_median1.get_medians()
print(22,clus_distance/22)

"""Thank You"""