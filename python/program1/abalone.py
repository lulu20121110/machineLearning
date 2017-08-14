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
        alpha-=0.00001
        weights+=alpha*deltws
    return weights

def stocgraddscent2(xArr,yArr,numIter=200):
    xmat = array(xArr)
    m, n = shape(xmat)
    weights = ones(n)
    for j in range(numIter):
        for i in range(m):
            alpha =4/(1+i+j)+0.001
            yHat = sum(xmat[i]*weights)
            deltws = xmat[i]*(yArr[i]-yHat)
            weights += alpha*deltws
    return weights


xArr,yArr=loadDataSet('C:\Users\lulu\Desktop\self\python\program1\ex2.txt')
xMat=mat(xArr)
yMat=mat(yArr)
ws=standRegres(xArr,yArr)
#ws1=gradDscent(xArr,yArr)
ws2=stocGradDscent(xArr[0:2000],yArr[0:2000],0.34769)
ws3=stocGradDscent1(xArr[0:2000],yArr[0:2000],0.0012,800)
ws4=aStocGradDscent(xArr[0:2000],yArr[0:2000],0.362)
ws6=stocgraddscent2(xArr[0:2000],yArr[0:2000],800)

'''
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
'''
'''
yHat5=lwlrTest(xArr,xArr,yArr,0.01)
strInd=xMat[:,1].argsort(0)
xSort=xMat[strInd][:,0,:]
ax.plot(xSort[:,1],yHat5[strInd],'k')
'''
'''
plt.show()
'''
'''
ws3=stocGradDscent1(xArr,yArr,0.0012,800);yHat_3=xMat*mat(ws3).T;corrcoef(yHat_3.flatten().A[0],yMat.T[:,0].flatten().A[0])
0.7178
ws=standRegres(xArr,yArr);yHat_=xMat*ws;corrcoef(yHat_.flatten().A[0],yMat.T[:,0].flatten().A[0])
0.7196
ws2=stocGradDscent(xArr,yArr,0.34769);yHat_2=xMat*mat(ws2).T;corrcoef(yHat_2.flatten().A[0],yMat.T[:,0].flatten().A[0])
0.65466405
ws4=aStocGradDscent(xArr,yArr,0.362);yHat_4=xMat*mat(ws4).T;corrcoef(yHat_4.flatten().A[0],yMat.T[:,0].flatten().A[0])
0.66009381参数del,alpha要再调大一点试试









'''