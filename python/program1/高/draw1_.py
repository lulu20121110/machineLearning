from numpy import *
import  matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

def readdata(filename):
    numfeat = len(open(filename).readline().split('\t'))-1
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = []
        curline = line.strip().split('\t')
        for i in range(numfeat):
            linearr.append(float(curline[i]))
        datamat.append(linearr)
        labelmat.append(float(curline[-1]))
    return datamat, labelmat
def stocgraddscent1(xArr,yArr,numIter=200):
    S=[]
    xmat = array(xArr)
    m, n = shape(xmat)
    alpha = 0.01
    weights = ones(n)
    for j in range(numIter):
        for i in range(m):
            yHat = sum(xmat[i]*weights)
            deltws = xmat[i]*(yArr[i]-yHat)
            weights += alpha*deltws
            yHat1 = sum(xmat[i] * weights)
        s = rssError(yArr, yHat1.T)
        S.append(s)
    ax.plot(range(numIter), S,'purple')
def rssError(yArr,yHatArr):
    a=sum((yArr-yHatArr)**2)
    return a
def graddscent(xArr,yArr):
    S=[]
    xmat = mat(xArr)
    ymat = mat(yArr).T
    m, n = shape(xmat)
    alpha = 0.01
    maxcycles = 200
    weights = ones((n, 1))
    for k in range(maxcycles):
        yHat = xmat*weights
        deltws = xmat.T*(ymat-yHat)
        weights += alpha*deltws/m
        yHat1 = xmat * weights
        s=rssError(ymat.A,yHat1.A)
        S.append(s)
    ax.plot(range(maxcycles),S,'red')
def stocgraddscent2(xArr,yArr,numIter=200):
    S=[]
    xmat = array(xArr)
    m, n = shape(xmat)
    weights = ones(n)
    for j in range(numIter):
        for i in range(m):
            alpha =4/(1+i+j)+0.001
            yHat = sum(xmat[i]*weights)
            deltws = xmat[i]*(yArr[i]-yHat)
            weights += alpha*deltws
            yHat2 = sum(xmat[i] * weights.T)
        s = rssError(yArr, yHat2.T)
        S.append(s)
    ax.plot(range(numIter), S, 'black')

abx, aby=readdata('ex2.txt')
stocgraddscent1(abx[0:3000],aby[0:3000])
graddscent(abx[0:3000],aby[0:3000])
stocgraddscent2(abx[0:3000],aby[0:3000])
plt.show()