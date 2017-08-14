# -*- coding: utf-8 -*-
'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)#求平均值
    meanRemoved = dataMat - meanVals #remove mean 去中心化
    covMat = cov(meanRemoved, rowvar=0)#协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))#特征值和特征向量，特征值越大，协方差越大
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions 逆序，从大到小
    redEigVects = eigVects[:,eigValInd]    #reorganize eig vects largest to smallest 变换矩阵
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions 样本数据在新特征空间的表示
    reconMat = (lowDDataMat * redEigVects.T) + meanVals#重构？？？？？？？？？？？
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
