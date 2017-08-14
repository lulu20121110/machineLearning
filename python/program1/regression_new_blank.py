'''
Created on Jan 8, 2011

@author: Peter
'''
from numpy import *

'''general function to parse tab -delimited floats'''
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
        diffMat =     
        weights[j,j] = 
    xTx = 
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = 
    return testPoint * ws

#loops over all the data points and applies lwlr to each one
def lwlrTest(testArr,xArr,yArr,k=1.0): 
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = 
    return yHat

