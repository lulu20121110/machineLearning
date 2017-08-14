from numpy import *
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

def standregres(xArr,yArr):
    xmat = mat(xArr)
    ymat = mat(yArr).T
    xTx = xmat.T * xmat
    if linalg.det(xTx)==0.0:
        print('ERROR')
    ws = xTx.I*(xmat.T*ymat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def graddscent(xArr,yArr):
    xmat = mat(xArr)
    ymat = mat(yArr).T
    m, n = shape(xmat)
    alpha = 0.001
    maxcycles = 200
    weights = ones((n, 1))
    for k in range(maxcycles):
        yHat = xmat*weights
        deltws = xmat.T*(ymat-yHat)
        weights += alpha*deltws/m
    return weights


def rssError(yArr,yHatArr):
    a=sum((yArr-yHatArr)**2)
    return a

def stocgraddscent1(xArr,yArr,numIter=200):
    xmat = array(xArr)
    m, n = shape(xmat)
    alpha = 0.001
    weights = ones(n)
    for j in range(numIter):
        for i in range(m):
            yHat = sum(xmat[i]*weights)
            yHat2 = sum(xmat[i]*weights.T)
            deltws = xmat[i]*(yArr[i]-yHat)
            weights += alpha*deltws
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


