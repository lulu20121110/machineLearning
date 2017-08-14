# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 12:38:36 2017

@author: lulu
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
iris=load_iris()
yTrain=list(iris.target[:30])+list(iris.target[50:80])
xTrain=list(iris.data[:30])+list(iris.data[50:80])
yTest=list(iris.target[30:50])+list(iris.target[80:100])
xTest=list(iris.data[30:50])+list(iris.data[80:100])
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(xTrain,yTrain)
clf.predict(xTest)


fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(range(len(xTest)),yTest)
ax.scatter(range(len(xTest)),list(clf.predict(xTest)))
plt.show()

print corrcoef(yTest,list(clf.predict(xTest)))









