# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 12:38:36 2017

@author: lulu
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris

#加载数据集，分别用于训练与测试
iris=load_iris()
yTrain=list(iris.target[:30])+list(iris.target[50:80])
xTrain=list(iris.data[:30])+list(iris.data[50:80])
yTest=list(iris.target[30:50])+list(iris.target[80:100])
xTest=list(iris.data[30:50])+list(iris.data[80:100])

#SVM分类器
clf = svm.SVC(kernel='sigmoid',gamma=0.01)
#训练
clf.fit(xTrain,yTrain)
#预测
clf.predict(xTest)


#画图
fig=plt.figure()
ax=fig.add_subplot(111)
ymajorLocator= MultipleLocator(1)
ax.yaxis.set_major_locator(ymajorLocator)
dot1=ax.scatter(range(len(xTest)),yTest,c='k',label='expected')
dot2=ax.scatter(range(len(xTest)),list(clf.predict(xTest)),c='r',marker='^',label='predicted')
plt.xlabel('No. of Sample')  
plt.ylabel('class') 
plt.legend()
plt.show()

print '相关系数:',corrcoef(yTest,list(clf.predict(xTest)))[0][1]

