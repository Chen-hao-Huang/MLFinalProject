#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:45:54 2017

@author: Chenhao
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


#Load train and test data
train = np.load("data/positions_data_npy/forwards.npy")

print(train[0,])
#Create train and test arrays
Xtr = train[:,0:-1]
Ytr = np.array(map(int, train[:,-1]))
print(Xtr[0,])
#
##Add your code below
#for i in range(1,41):
#    kmeans=KMeans(n_clusters = i)
#    Z = kmeans.fit_predict(Xtr)
#    quality = cluster_utils.cluster_quality(Xtr,Z,i)
#    
#    
#K = 30
#kmeans=KMeans(n_clusters = K)
#Z = kmeans.fit_predict(Xtr)
#p = cluster_utils.cluster_proportions(Z,K)
##for i in p:
##    print i
#    
#cluster_utils.show_means(cluster_utils.cluster_means(Xtr,Z,K),p)
#
#
#
#for i in range(1,41):
#     kmeans=cluster_class.cluster_class( i)
#     Z = kmeans.fit(Xtr,Ytr)
#     Z1 = kmeans.predict(Xte)
#     print kmeans.score(Xte,Yte)
     
     