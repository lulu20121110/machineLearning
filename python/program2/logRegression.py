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

def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(xArr,yArr,alpha=0.001,maxCycles=500):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    m,n=shape(xMat)
    weights=ones((n,1))
    for k in range(maxCycles):
        yHat=xMat*weights
        h=sigmoid(yHat)
        deltws=xMat.T*(yMat-h)
        weights+=alpha*deltws
    return weights

def stocGradAscent1(xArr,yArr,alpha=0.001,numIter=200):
    xMat=array(xArr)
    m,n=shape(xMat)
    
    weights=ones(n)
    for j in range(numIter):
        for k in range(m):
            yHat=sum(xMat[k]*weights)
            h=sigmoid(yHat)
            deltws=(yArr[k]-h)*xMat[k]
            weights+=alpha*deltws
    return weights



def plotBestFit(w,filename='C:\Users\lulu\Desktop\self\python\program2\horseColicTest.txt'):
    weights=w
    xMat,yMat=loadDataSet(filename)
    import pca
    lowDDataMat, reconMat=pca.pca(xMat,2)
    dataArr=array(lowDDataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(yMat[i]==1):
            xcord1.append(dataArr[i,0])
            ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0])
            ycord2.append(dataArr[i,1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,c='red')
    ax.scatter(xcord2,ycord2,c='green')
    x=arange(-80,80,1)
    y=-weights[0]*x/weights[1]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.xlabel('x2')
    plt.show()
    return testLogRegres(weights,lowDDataMat,yMat)
    
def testLogRegres(weights, test_x, test_y):  
    numSamples, numFeatures = shape(test_x)  
    matchCount = 0  
    for i in xrange(numSamples):  
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5  
        if predict == bool(test_y[i]):  
            matchCount += 1  
    accuracy = float(matchCount) / numSamples  
    return accuracy

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5: return 1.0
    else: return 0.0
    
def colicTest():
    frTrain=open('C:\Users\lulu\Desktop\self\python\program2\horseColicTraining.txt')
    frTest=open('C:\Users\lulu\Desktop\self\python\program2\horseColicTest.txt')
    dataMat=[];labelMat=[]
    for line in frTrain.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(21):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    trainWeights=stocGradAscent1(array(dataMat),labelMat)
    errCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        curLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(curLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(curLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print 'the error rate of this test is:%f'%errorRate
    return errorRate

def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print 'after %d iterations the average error rate is: %f' %(numTests,errorSum/float(numTests))
            
    
xArr,yArr=loadDataSet('C:\Users\lulu\Desktop\self\python\program2\horseColicTraining.txt')
import pca
lowDDataMat, reconMat=pca.pca(xArr,2)
#plotBestFit(gradAscent(lowDDataMat,yArr))
'''
ws=gradAscent(lowDDataMat,yArr)

xMat=mat(lowDDataMat)
yMat=mat(yArr)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,0].flatten().A[0],xMat[:,1].flatten().A[0])
plt.show()
'''








'''


import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()
'''




