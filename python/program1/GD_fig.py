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


def aStocGradDscent(xArr,yArr):
    xMat=array(xArr)
    m,n=shape(xMat)
    alpha=0.2
    weights=ones(n)
    for k in range(m):
        yHat=sum(xMat[k]*weights)
        deltws=(yArr[k]-yHat)*xMat[k]
        alpha-=0.0001
        weights+=alpha*deltws
    return weights



xArr,yArr=loadDataSet('C:\Users\lulu\Desktop\self\python\ex0.txt')
xMat=mat(xArr)
yMat=mat(yArr)

ws=standRegres(xArr,yArr)
ws1=gradDscent(xArr,yArr)
ws2=stocGradDscent(xArr,yArr)
ws3=stocGradDscent1(xArr,yArr,500)
ws4=aStocGradDscent(xArr,yArr)

print ws
print ws1
print ws2
print ws3
print ws4


import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat,'r')
yHat1=xCopy*ws1
ax.plot(xCopy[:,1],yHat1,'k.')
yHat2=xCopy*mat(ws2).T
ax.plot(xCopy[:,1],yHat2,'k')
yHat3=xCopy*mat(ws3).T
ax.plot(xCopy[:,1],yHat3,'y')
yHat4=xCopy*mat(ws4).T
ax.plot(xCopy[:,1],yHat4,'g')


plt.show()





