# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 21:55:52 2017

@author: lulu
"""

import matplotlib
import matplotlib.pyplot as plt
import pca

dataMat=pca.loadDataSet('C:\Users\lulu\Desktop\self\python\program2\iris.txt')

idx_0 = np.where(labels==0)
idx_1 = np.where(labels==1)
idx_2 = np.where(labels==2)


lowDDataMat, reconMat=pca.pca(dataMat,2)

fig=plt.figure()
ax1=fig.add_subplot(311)
ax1.scatter(dataMat[idx_0[0],0].flatten().A[0],dataMat[idx_0[0],1].flatten().A[0],c='r')
ax1.scatter(dataMat[idx_1[0],0].flatten().A[0],dataMat[idx_1[0],1].flatten().A[0],c='g')
ax1.scatter(dataMat[idx_2[0],0].flatten().A[0],dataMat[idx_2[0],1].flatten().A[0],c='b')
ax2=fig.add_subplot(312)
ax2.scatter(reconMat[idx_0[0],0].flatten().A[0],reconMat[idx_0[0],1].flatten().A[0],c='r')
ax2.scatter(reconMat[idx_1[0],0].flatten().A[0],reconMat[idx_1[0],1].flatten().A[0],c='g')
ax2.scatter(reconMat[idx_2[0],0].flatten().A[0],reconMat[idx_2[0],1].flatten().A[0],c='b')
ax3=fig.add_subplot(313)
ax3.scatter(lowDDataMat[idx_0[0],0].flatten().A[0],lowDDataMat[idx_0[0],1].flatten().A[0],c='r')
ax3.scatter(lowDDataMat[idx_1[0],0].flatten().A[0],lowDDataMat[idx_1[0],1].flatten().A[0],c='g')
ax3.scatter(lowDDataMat[idx_2[0],0].flatten().A[0],lowDDataMat[idx_2[0],1].flatten().A[0],c='b')
plt.show()

'''
lowDDataMat, reconMat=pca.pca(dataMat,1)

fig=plt.figure()
ax1=fig.add_subplot(311)
ax1.scatter(dataMat[idx_0[0],0].flatten().A[0],dataMat[idx_0[0],1].flatten().A[0],c='r')
ax1.scatter(dataMat[idx_1[0],0].flatten().A[0],dataMat[idx_1[0],1].flatten().A[0],c='g')
ax1.scatter(dataMat[idx_2[0],0].flatten().A[0],dataMat[idx_2[0],1].flatten().A[0],c='b')
ax2=fig.add_subplot(312)
ax2.scatter(reconMat[idx_0[0],0].flatten().A[0],reconMat[idx_0[0],1].flatten().A[0],c='r')
ax2.scatter(reconMat[idx_1[0],0].flatten().A[0],reconMat[idx_1[0],1].flatten().A[0],c='g')
ax2.scatter(reconMat[idx_2[0],0].flatten().A[0],reconMat[idx_2[0],1].flatten().A[0],c='b')
ax3=fig.add_subplot(313)
ax3.scatter(lowDDataMat[idx_0[0],0].flatten().A[0],zeros(50),c='r')
ax3.scatter(lowDDataMat[idx_1[0],0].flatten().A[0],zeros(50),c='g')
ax3.scatter(lowDDataMat[idx_2[0],0].flatten().A[0],zeros(50),c='b')
plt.show()
'''