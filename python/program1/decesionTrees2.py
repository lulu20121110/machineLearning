# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 14:11:17 2017

@author: lulu
"""

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

def createDataSet(filename):  
    dataSet =loadDataSet(filename)  
    features = ['age of the patient','spectacle prescription','astigmatic','tear production rate']  
    return dataSet,features 

def treeGrowth(dataSet,features):
    classList = [example[-1] for example in dataSet]  
    if classList.count(classList[0])==len(classList):  #only one decision
        print ' h h h'###############leaf node
        print classList[0]
        return classList[0]  
    if len(dataSet[0])==1:# no features left
        return classify(classList)  
  
    bestFeat = findBestSplit(dataSet)#bestFeat is the index of best feature
    print 'bestFeat'+str(bestFeat)########
    bestFeatLabel = features[bestFeat]  
    myTree = {bestFeatLabel:{}}  
    featValues = [example[bestFeat] for example in dataSet]  
    uniqueFeatValues = set(featValues)    
    for values in uniqueFeatValues:  
        subDataSet = splitDataSet(dataSet,bestFeat,values)  
        print 'xxx'###############recursion
        myTree[bestFeatLabel][values] = treeGrowth(subDataSet,features)  
    print 'lll'###############end loop
    global bestFeat1############
    bestFeat1=bestFeat############
    print bestFeat1############
    #del (features[bestFeat])###############
    return myTree

#there is an error below
def classify(classList):  
    ''''' 
    find the most in the set 
    '''  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys():  
            classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)  
    return sortedClassCount[0][0]  

def findBestSplit(dataset):  
    numFeatures = len(dataset[0])-1  
    baseEntropy = calcShannonEnt(dataset) 
    bestInfoGain = 0.0  
    bestFeat = -1#####  
    for i in range(numFeatures):  
        featValues = [example[i] for example in dataset]  
        uniqueFeatValues = set(featValues)  
        newEntropy = 0.0  
        for val in uniqueFeatValues:  
            subDataSet = splitDataSet(dataset,i,val)  
            prob = len(subDataSet)/float(len(dataset))  
            newEntropy += prob*calcShannonEnt(subDataSet) 
        print str(i)+' '+str(newEntropy)############
        if(baseEntropy - newEntropy)>bestInfoGain:  
            bestInfoGain = baseEntropy - newEntropy  
            bestFeat = i  
    return bestFeat

def splitDataSet(dataset,feat,values):  
    retDataSet = []  
    for featVec in dataset:  
        if featVec[feat] == values:  
            reducedFeatVec = featVec[:feat]  
            reducedFeatVec.extend(featVec[feat+1:])  
            retDataSet.append(reducedFeatVec)
    return retDataSet

def calcShannonEnt(dataset):  
    numEntries = len(dataset)  
    labelCounts = {}  
    for featVec in dataset:  
        currentLabel = featVec[-1]  
        if currentLabel not in labelCounts.keys():  
            labelCounts[currentLabel] = 0  
        labelCounts[currentLabel] += 1  
    shannonEnt = 0.0  
  
    for key in labelCounts:  
        prob = float(labelCounts[key])/numEntries  
        if prob != 0:  
            shannonEnt -= prob*log(prob,2)  
    return shannonEnt

def predict(tree,newObject):  
    while isinstance(tree,dict):  
        key = tree.keys()[0]  
        tree = tree[key][newObject[key]]  
    return tree  
  
if __name__ == '__main__':  
    dataset,features = createDataSet('C:\Users\lulu\Desktop\self\python\lenses.txt')  
    tree = treeGrowth(dataset,features)  
    print tree 
    '''
#1  	1  	1  	1  	3
 #1  	1   	1  	2  	2
#1 	1  	2  	1  	3
1  	1  	2  	2  	1
#1  	2  	1  	1  	3
 #1  	2   	1  	2 	2
#1  	2  	2  	1  	3
1 	2  	2  	2  	1
#2  	1  	1  	1  	3
 #2  	1   	1  	2  	2
#2  	1  	2  	1  	3
2  	1  	2  	2  	1
#2  	2  	1  	1  	3
 #2  	2   	1  	2  	2
#2  	2  	2  	1  	3
2  	2  	2  	2  	3
#3  	1  	1  	1  	3
#3  	1   	1  	2  	3
#3  	1  	2  	1  	3
3  	1  	2  	2  	1
#3  	2  	1  	1  	3
#3  	2   	1  	2  	2
#3  	2  	2  	1  	3
3  	2  	2  	2  	3
'''