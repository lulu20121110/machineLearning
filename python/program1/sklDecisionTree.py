# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 12:58:29 2017

@author: lulu
"""

from sklearn import tree
from numpy import *
import operator
from math import log

def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))
    dataSet=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataSet.append(lineArr)
    return dataSet

def loadDataSet1(filename):
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

def createDataSet(filename):  
    dataSet =loadDataSet(filename)  
    features = ['age of the patient','spectacle prescription','astigmatic','tear production rate']  
    return dataSet,features 

if __name__ == '__main__':  
    xArr,yArr=loadDataSet1('C:\Users\lulu\Desktop\self\python\program1\lenses.txt')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(xArr[0:18],yArr[0:18])
    clf.predict(xArr[18:24])
    