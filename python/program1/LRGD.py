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



def gradDscent(xArr,yArr,maxCycles=500):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m,n=shape(xMat)
    alpha=0.001
    weights=ones((n,1))
    for k in range(maxCycles):
        yHat=xMat*weights
        deltws=xMat.T*(yMat-yHat)
        weights+=alpha*deltws
    return weights

def stocGradDscent(xArr,yArr,alpha=0.001):
    xMat=array(xArr)
    m,n=shape(xMat)
    
    weights=ones(n)
    for k in range(m):
        yHat=sum(xMat[k]*weights)
        deltws=(yArr[k]-yHat)*xMat[k]
        weights+=alpha*deltws
    return weights

def stocGradDscent1(xArr,yArr,alpha=0.001,numIter=200):
    xMat=array(xArr)
    m,n=shape(xMat)
    
    weights=ones(n)
    for j in range(numIter):
        for k in range(m):
            yHat=sum(xMat[k]*weights)
            deltws=(yArr[k]-yHat)*xMat[k]
            weights+=alpha*deltws
    return weights


def aStocGradDscent(xArr,yArr,alpha=0.2):
    xMat=array(xArr)
    m,n=shape(xMat)
    
    weights=ones(n)
    for k in range(m):
        yHat=sum(xMat[k]*weights)
        deltws=(yArr[k]-yHat)*xMat[k]
        alpha-=0.0001
        weights+=alpha*deltws
    return weights