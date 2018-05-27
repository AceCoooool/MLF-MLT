---
title: 机器学习技法作业2：part2
date: 2017-02-17 19:28:21
tags: MLF&MLT
categories: ML
---

机器学习技法课后作业2-1：对应题目12～题目20
<!-- more -->

## 机器学习技法作业2

## 问题12-18

通过PPT208介绍的adaboost算法，实现一个AdaBoost-Stump算法。并在给定的训练集和测试集上进行运行。
采用$T=300$次迭代（即采用300个决策树桩函数），最终的$E_{in}$和$E_{out}$采用$0/1$误差

决策树桩的实现可根据下述的步骤进行：
① 对于任意特征$i$，可以先对该维度数据$x_{n,i}$进行排序，排序后满足：$x_{|n|,i}\le x_{|n+1|,i}$
② 考虑截距从$-\infty$以及所有的中点$(x_{|n|,i}+x_{|n+1|,i})/2$，并结合$s=\{+1,-1\}$通过最小化该维度$i$对应的最小$E_{in}^u$来获得最佳的$(s,\theta)$
③ 针对所有的维度，选取最佳的$(s,i,\theta)$

```python
# 导入库
import numpy as np
import pandas as pd
import scipy.linalg as lin
```
```python
# 加载数据函数
def loadData(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X = data[:, 0: row-1]
    Y = data[:, row-1:row]
    return X, Y
```
```python
# 决策树桩
def decision_stump(X, Y, thres, U):
    row, col = X.shape
    r, c = thres.shape; besterr = 1
    btheta = 0; bs = 0; index = 0
    for i in range(col):
        Yhat1 = np.sign(np.tile(X[:, i:i+1], (1, r)).T-thres[:, i:i+1]).T
        err1 = (Yhat1!=Y).T.dot(U)
        err2 = (-1*Yhat1!=Y).T.dot(U)
        s = 1 if np.min(err1) < np.min(err2) else -1
        if s == 1 and np.min(err1) < besterr:
            besterr = np.min(err1); bs = 1
            index = i; btheta = thres[np.argmin(err1), i]
        if s == -1 and np.min(err2) < besterr:
            besterr = np.min(err2); bs = -1
            index = i; btheta = thres[np.argmin(err2), i]
    return besterr, btheta, bs, index
```
```python
# AdaBoost---Stump 算法
# 需要说明: 与PPT上有点不同，始终保证sum(U)=1
def ada_boost(X, Y, T):
    row, col = X.shape
    U = np.ones((row, 1))/row
    Xsort = np.sort(X, 0)
    thres = (np.r_[Xsort[0:1, :] - 0.1, Xsort[0:row - 1, :]] + Xsort) / 2
    theta = np.zeros((T,)); s = np.zeros((T,));
    index = np.zeros((T,)).astype(int); alpha = np.zeros((T,))
    err = np.zeros((T,))
    for i in range(T):
        err[i], theta[i], s[i], index[i] = decision_stump(X, Y, thres, U)
        yhat = s[i]*np.sign(X[:, index[i]:index[i]+1]-theta[i])
        delta = np.sqrt((1-err[i])/err[i])
        U[yhat==Y] /= delta
        U[yhat!=Y] *= delta
# Q14运行时，解除注释
#        if i == T-1:
#            print('sum(U): ', np.sum(U))
        alpha[i] = np.log(delta)
        U /= np.sum(U)
# Q15运行时，解除注释
#    print('最小的eta: ', np.min(err))
    return theta, index, s, alpha
```
```python
# 预测函数
def predict(X, theta, index, s, alpha):
    row, col = X.shape
    num = len(theta)
    ytemp = np.tile(s.reshape((1, num)), (row, 1))*np.sign(X[:, index]-theta.reshape((1, num)))
    yhat = np.sign(ytemp.dot(alpha.reshape(num, 1)))
    return yhat
```
```python
# 导入数据
X, Y = loadData('hw2_adaboost_train.dat')
Xtest, Ytest = loadData('hw2_adaboost_test.dat')
row, col = X.shape
r, c = Xtest.shape
```
### 问题12

Q12：$E_{in}(g_1)$的结果为多少？

A12：令T=1

```python
# Q12
theta, index, s, alpha = ada_boost(X, Y, 1)
Ypred = predict(X, theta, index, s, alpha)
print('Ein(g1)：', np.sum(Ypred!=Y)/row)
```

    Ein(g1)： 0.24
### 问题13

Q13：$E_{in}(G)$的结果为多少？

A13：T=300

```python
# Q13
theta, index, s, alpha = ada_boost(X, Y, 300)
Ypred = predict(X, theta, index, s, alpha)
print('Ein(G)：', np.sum(Ypred!=Y)/r)
```

    Ein(G)： 0.0
### 问题14

Q14：令$U_t=\sum_{n=1}^Nu_n^{(t)}$，$U_2$的值是多少（$U_1=1$）？

A14：

```python
# Q14 --- 打开上述注释项，在运行一次
theta, index, s, alpha = ada_boost(X, Y, 1)
```

    sum(U):  0.854166260163
### 问题15

Q15：$U_T$的值是多少？

A15：该问题采用“和为1”化后无法给出理想答案，但是取消“和为1”的条件，则会导致程序跑崩。希望知道如何改善的给点提示

### 问题16

Q16：对于$t=1,2,...,300$的所有$\epsilon_t$中，最小的值为多少？

A16：

```python
# Q16 
theta, index, s, alpha = ada_boost(X, Y, 300)
```

    最小的epsilon:  0.178728070175

### 问题17

Q17：对测试集计算$E_{out}$，对应的$E_{out}(g_1)$为多少？

A17：

```python
# Q17
theta, index, s, alpha = ada_boost(X, Y, 1)
Ypred = predict(Xtest, theta, index, s, alpha)
print('Eout(g1)：', np.sum(Ypred!=Ytest)/r)
```

    Eout(g1)： 0.29

### 问题18

Q18：对测试集计算$E_{out}$，对应的$E_{out}(G)$为多少？

A18：

```python
# Q18
theta, index, s, alpha = ada_boost(X, Y, 300)
Ypred = predict(Xtest, theta, index, s, alpha)
print('Eout(G)：', np.sum(Ypred!=Ytest)/r)
```

    Eout(G)： 0.132

## 问题19-20

根据Lec206实现一个kernel ridge regression算法，并将其运用到分类问题中。根据给定的数据集，取其前400个样本作为训练集，剩余的样本作为测试集。利用$0/1$误差计算$E_{in}$和$E_{out}$。考虑采用高斯核$exp(-\gamma ||x-x^\prime||^2)$，并尝试下述所有的参数$\gamma\in\{32,2,0.125\}$和$\lambda\in\{0.001,1,1000\}$

```python
# ----------- Q19-20 --------------
# 获得对偶矩阵K
def matK(X, X1, gamma):
    row, col =X.shape
    r, c = X1.shape
    K = np.zeros((row, r))
    for i in range(r):
        K[:, i] = np.sum((X-X1[i:i+1, :])**2, 1)
    K = np.exp(-gamma*K)
    return K
```
```python
# 加载数据
X, Y = loadData('hw2_lssvm_all.dat')
Xtrain = X[0:400, :]; Ytrain = Y[0:400, :]
Xtest = X[400:, :]; Ytest = Y[400:, :]
row, col = Xtest.shape
```
### 问题19&问题20

Q19&Q20：以上给出的所有参数组合中，最小的$E_{in}(g)$和最小的$E_{out}(g)$分别是多少？

A19&A20：

```python
# 测试
gamma = [32, 2, 0.125]
lamb = [0.001, 1, 1000]
Ein = np.zeros((len(gamma), len(lamb)))
Eout = np.zeros((len(gamma), len(lamb)))
for i in range(len(gamma)):
    K = matK(Xtrain, Xtrain, gamma[i])
    K2 = matK(Xtrain, Xtest, gamma[i])
    for j in range(len(lamb)):
        beta = lin.pinv(lamb[j]*np.eye(400)+K).dot(Ytrain)
        yhat = np.sign(K.dot(beta))
        Ein[i, j] = np.sum(yhat != Ytrain)/400
        yhat2 = np.sign(K2.T.dot(beta))
        Eout[i, j] = np.sum(yhat2 != Ytest)/row
print('最小的Ein: ', np.min(Ein))
print('最小的Eout: ', np.min(Eout))
```

    最小的Ein:  0.0
    最小的Eout:  0.39
