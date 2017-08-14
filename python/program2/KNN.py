# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 09:29:53 2017

@author: lulu
"""

from numpy import *
import operator
from sklearn.datasets import load_iris

##给出训练数据以及对应的类别
def createDataSetIris():
    iris=load_iris()
    group = iris.data
    labels = iris.target
    return group,labels

def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))##建立与dataSet结构一样的矩阵
    m = dataSet.shape[0]
    for i in range(1,m):
        normDataSet[i,:] = (dataSet[i,:] - minVals) / ranges
    return normDataSet


###通过KNN进行分类
def classify(inX,dataSet,label,k):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    diff = tile(inX,(dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff,axis = 1)###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    
    ##对距离进行排序
    sortedDistIndex = argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes

def knngroup(dataMat1,dataMat,labelMat,k):
    dataMat1=mat(dataMat1)
    m=shape(dataMat1)[0]
    for i in range(m):
        pre[i]=classify(dataMat1[i,:],dataMat,labelMat,k)
    return pre

group,labels=createDataSetIris()

import pca
lowDDataMat, reconMat=pca.pca(group_norm,2)
group_norm=autoNorm(lowDDataMat)

'''
#对样本点画图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(group_norm[:,0],group_norm[:,1],c=15.0*array(labels))
plt.show()


#降维后画图
import pca
lowDDataMat, reconMat=pca.pca(group,2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(reconMat.A[:,0],reconMat.A[:,1],c=15.0*array(labels))
plt.show()


#全部填充
xx,yy = np.meshgrid(np.arange(0,1,0.1), np.arange(0,1,0.1))
coords = np.c_[xx.ravel(), yy.ravel()]
pre=[]
m=shape(coords)[0]
for i in range(m):
    pre.append(classify(coords[i,:],group_norm,labels,k))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(group_norm[:,0],group_norm[:,1],c=15.0*array(labels))
ax.scatter(coords[:,0],coords[:,1],c=10.0*(array(pre)+0.001))
plt.show()

#或
pre=array(pre).reshape(xx.shape)
plt.pcolormesh(xx, yy, pre, cmap=plt.cm.Paired)
plt.pcolormesh(xx, yy, pre, cmap=plt.cm.coolwarm)

#准确率
right=0
pre=[]
for i in range(150):
    pre.append(classify(group_norm[i,:],group_norm,labels,1))
    if array(pre)[i]==labels[i]:
        right+=1
print float(right)/150
'''