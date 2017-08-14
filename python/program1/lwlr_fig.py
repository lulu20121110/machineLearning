# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 21:45:28 2017

@author: lulu
"""

from numpy import *

def loadDataSet(fileName):     
    #get number of fields
    numFeat = len(open(fileName).readline().split('\t')) - 1 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    #next 2 lines create weights matrix
    for j in range(m):                     
        diffMat =testPoint - xMat[j, :]
        weights[j,j] =exp(diffMat * diffMat.T/(-2.0 * k**2)) 
    xTx = xMat.T*(weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint * ws

#loops over all the data points and applies lwlr to each one
def lwlrTest(testArr,xArr,yArr,k=1.0): 
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

import matplotlib.pyplot as plt
xArr,yArr=loadDataSet('C:\Users\lulu\Desktop\self\python\ex0.txt')
yHat=lwlrTest(xArr,xArr,yArr,0.01)
xMat=mat(xArr)
strInd=xMat[:,1].argsort(0)
xSort=xMat[strInd][:,0,:]
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[strInd])
ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0])
plt.show()