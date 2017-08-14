# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:54:06 2017

@author: lulu
"""
from numpy import *

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

def standRegres(xArr,yArr):
	xMat=mat(xArr)
	yMat=mat(yArr).T
	xTx=xMat.T*xMat
	if linalg.det(xTx)==0.0:
		print"error"
		return
	ws=xTx.I*(xMat.T*yMat)
	return ws

def gradDscent(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m,n=shape(xMat)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        yHat=xMat*weights
        deltws=xMat.T*(yMat-yHat)
        weights+=alpha*deltws
    return weights

def stocGradDscent(xArr,yArr):
    xMat=array(xArr)
    m,n=shape(xMat)
    alpha=0.001
    weights=ones(n)
    for k in range(m):
        yHat=sum(xMat[k]*weights)
        deltws=(yArr[k]-yHat)*xMat[k]
        weights+=alpha*deltws
    return weights

def stocGradDscent1(xArr,yArr,numIter=200):
    xMat=array(xArr)
    m,n=shape(xMat)
    alpha=0.001
    weights=ones(n)
    for j in range(numIter):
        for k in range(m):
            yHat=sum(xMat[k]*weights)
            deltws=(yArr[k]-yHat)*xMat[k]
            weights+=alpha*deltws
    return weights


xArr,yArr=loadDataSet('C:\Users\lulu\Desktop\self\python\ex0.txt')
ws=standRegres(xArr,yArr)
xMat=mat(xArr)
yMat=mat(yArr)
yHat=xMat*ws
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()

ws1=gradDscent(xArr,yArr)
print ws1

ws2=stocGradDscent(xArr,yArr)
print ws2

ws3=stocGradDscent1(xArr,yArr)
print ws3