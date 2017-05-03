#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:45:54 2017

@author: Chenhao
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import DistanceMetric



#Load train and test data
train = np.load("data/positions_data_npy/midfielders.npy")

#Create train and test arrays
np.random.seed(1)
np.random.shuffle(train)
X = train[:,0:-1]
Ytr = np.array(map(int, train[:,-1]))
out = []

#V=np.cov(X)
#inv_covariance_xy = np.linalg.inv(V)
#xy_mean = np.mean(X,axis=0)
#print xy_mean.shape
#for i in range(0,len(X)):
#    for x_i in X:
##        x_diff = np.array([x_i - xy_mean[i]])

#mahalanobi = DistanceMetric.get_metric('mahalanobis',V = np.cov(X))
#metric='mahalanobis',metrics_params={'V' :np.cov(X)}
db = DBSCAN(eps=30000000, min_samples=30).fit(X)
labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

outliers = []
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 4], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 4], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
for i in range(0,labels.size):
    if labels[i] == -1:
        outliers.append(train[i])
        

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

for i in range(0,labels.size):
    if labels[i] == -1:
        outliers.append(train[i])
        if labels[i] == -1:
            plt.plot(train[i][4],i,'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=10)
        print((train[i])[4])

plt.show()
                
db = DBSCAN(eps=25000000, min_samples=40).fit(X)
labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 4], xy[:, 7], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 4], xy[:, 7], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

outliers = []
for i in range(0,labels.size):
    if labels[i] == -1:
        outliers.append(train[i])
        
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


kmeans=KMeans(n_clusters = 2)
Z = kmeans.fit_predict(outliers)
for i in range(0,Z.size):
     if Z[i] == 1:
        plt.plot(outliers[i][4],i,'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)
        print((outliers[i])[0])
print(np.sum(Z))

plt.show()



traindata = np.load("data/positions_data_npy/forwards.npy")

TrainErr = []
ValidateErr = []
np.random.seed(1)
np.random.shuffle(traindata)
m = traindata.shape[0]
train_end = int(0.5 * m)
train = traindata[:train_end]
train_x = train[:, 8:]
train_y = train[:, 7]
train_z = train[:, 4]
validate = traindata[train_end:]
validate_x = validate[:, 8:]
validate_y  = validate[:, 4]
maxParameter = 0 

lr = LinearRegression()
predicted = lr.fit(train_x,train_z)
predicted = lr.predict(validate_x)

fig, ax = plt.subplots()
ax.scatter(validate_y, predicted)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()




#Load train and test data
train = np.load("data/forwards.npy")

#Create train and test arrays
np.random.seed(1)
np.random.shuffle(train)
X = train[:,7:-1]
Ytr = np.array(map(int, train[:,-1]))
out = []


db = DBSCAN(eps=55, min_samples=50).fit(X)
labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

outliers = []
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    class_member_mask = (labels == k)
    xy = train[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 4], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)
    xy = train[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 4], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
for i in range(0,labels.size):
    if labels[i] == -1:
        outliers.append(train[i])
        

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

