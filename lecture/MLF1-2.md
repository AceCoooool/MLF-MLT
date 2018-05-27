---
title: 机器学习基石作业1：part2
date: 2017-02-06 20:50:09
tags: MLF&MLT
categories: ML
---

机器学习基石课后作业1-2：对应题目15～题目20
<!-- more -->

## 机器学习基石作业1

## 问题6

Q15~Q20：主要考察感知机算法和Pocket算法。以下给出两种算法的“口语化”说明：

### 算法说明

PLA算法

>初始化$w=0$
>For t=0,1,...
>​	① 寻找使得$sign(w_t^Tx_{n(t)})\neq y_{n(t)}$的数据$(x_{n(t)}, y_{n(t)})$  
>​	② 按照$w_{t+1}\leftarrow w_t+y_{n(t)}x_{n(t)} $ 修正$w$
>循环直到没有错误的情况为止

其中①中有些注意事项：可以采用不同的策略，如 (a).每次从前往后$(1,...,N)$寻找“错误数据” (b). 每次从前一次“错误数据”开始往后寻找“错误数据” (c). 每次随机打乱数据，按打乱后的顺序从前往后寻找“错误数据”。  相比较而言，方案(b)(c)要更加快速。

但PLA算法只能针对线性可分的数据，因此引入Pocket算法。

Pocket算法

>初始化$w=0$
>For t=0,1,...
>​	①寻找使得$sign(w_t^Tx_{n(t)})\neq y_{n(t)}$的数据$(x_{n(t)}, y_{n(t)})$
>​	②按照$w_{t+1}\leftarrow w_t+y_{n(t)}x_{n(t)} $ ，(试图)修正错误
>​	③如果$w_{t+1}$的错误率小于$w_t$，则令$w=w_{t+1}$
>直到预先设定的循环次数
>返回$w$

其中①中寻找“错误数据”大多数情况下采用(c)：编程实现则直接找出所有错误，再随便从中随机选一个。

### 算法实现

```python
# 感知机算法
# 下述实现需注意：加入了一个prevpos变量，为了保证每次都先从当前数据的后面数据中寻找错误项
#（这样的方式相比每次均从第一个数据开始寻找要更快速）
def perceptron(X, Y, theta, eta=1):
    num = 0; prevpos = 0
    while(True):
        yhat = np.sign(X.dot(theta))
        yhat[np.where(yhat == 0)] = -1
        index = np.where(yhat != Y)[0]
        if not index.any():
            break
        if not index[index >= prevpos].any():
            prevpos = 0
        pos = index[index >= prevpos][0]
        prevpos = pos
        theta += eta*Y[pos, 0]*X[pos:pos+1, :].T
        num += 1
    return theta, num 
```
```python
# 在定义Pocket算法前，先引入错误率函数
def mistake(yhat, y):
    row, col = y.shape
    return np.sum(yhat != y)/row
```
```python
# Pocket算法
def pocket(X, Y, theta, iternum, eta = 1):
    yhat = np.sign(X.dot(theta))
    yhat[np.where(yhat == 0)] = -1
    errold = mistake(yhat, Y)
    thetabest = np.zeros(theta.shape)
    for t in range(iternum):
        index = np.where(yhat != Y)[0]
        if not index.any():
            break
        pos = index[np.random.permutation(len(index))[0]]
        theta += eta * Y[pos, 0] * X[pos:pos + 1, :].T
        yhat = np.sign(X.dot(theta))
        yhat[np.where(yhat == 0)] = -1
        errnow = mistake(yhat, Y)
        if errnow < errold:
            thetabest = theta.copy() # 这一步切勿弄错，如果直接thetabest=theta则会使两者指向同一块空间
            errold = errnow
    return thetabest, theta
```
### 具体问题

数据导入模块

```python
# 导入数据函数
def loadData(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X = np.c_[np.ones((col, 1)), data[:, 0: row-1]]
    Y = data[:, row-1:row]
    return X, Y
```
**Q15-17数据导入**
```python
# Q15-Q17导入数据项
X, Y = loadData('hw1_15_train.dat')
col, row = X.shape
theta = np.zeros((row, 1))
print('X的前五项：\n',X[0:5, :])
print('Y的前五项: \n',Y[0:5,:].T)
```

    X的前五项：
     [[ 1.        0.97681   0.10723   0.64385   0.29556 ]
     [ 1.        0.67194   0.2418    0.83075   0.42741 ]
     [ 1.        0.20619   0.23321   0.81004   0.98691 ]
     [ 1.        0.51583   0.055814  0.92274   0.75797 ]
     [ 1.        0.70893   0.10836   0.33951   0.77058 ]]
    Y的前五项: 
     [[ 1.  1.  1.  1.  1.]]
Q15：基础感知机算法更新次数

```python
# Q15的结果
theta, num = perceptron(X, Y, theta)
print('总共更新theta的次数：',num)
```

    总共更新theta的次数： 39

Q16：随机排列后感知机算法的平均更新次数

```python
# Q16的结果
total = 0
for i in range(2000):
    theta = np.zeros((row, 1))
    randpos = np.random.permutation(col)
    Xrnd = X[randpos, :]
    Yrnd = Y[randpos, 0:1]
    _, num = perceptron(Xrnd, Yrnd, theta)
    total += num
print('2000次平均每次更新theta的次数：',total/2000)
```

    2000次平均每次更新theta的次数： 39.806

Q17：不同$\eta$情况下的感知机平均更新次数

```python
# Q17的结果
total = 0
for i in range(2000):
    theta = np.zeros((row, 1))
    randpos = np.random.permutation(col)
    Xrnd = X[randpos, :]
    Yrnd = Y[randpos, 0:1]
    _, num = perceptron(Xrnd, Yrnd, theta, 0.5)
    total += num
print('2000次平均每次更新theta的次数：',total/2000)
```

    2000次平均每次更新theta的次数： 39.758

这里需要说明一点：Q17和Q16的结果基本一致的原因在于参数同时缩放对$sign(w^Tx)$来说是一样的，但当初始$w\neq 0$时，两问的结果还是有一定差别的

**Q18-Q20数据导入**

```python
# Q18-20导入数据
X, Y = loadData('hw1_18_train.dat')
Xtest, Ytest = loadData('hw1_18_test.dat')
col, row = X.shape
theta = np.zeros((row, 1))
```
Q18：50次更新情况下的测试集错误率

```python
# Q18
total = 0
for i in range(2000):
    theta = np.zeros((row, 1))
    randpos = np.random.permutation(col)
    Xrnd = X[randpos, :]
    Yrnd = Y[randpos, 0:1]
    theta, thetabad = pocket(Xrnd, Yrnd, theta, 50)
    yhat = np.sign(Xtest.dot(theta))
    yhat[np.where(yhat == 0)] = -1
    err = mistake(yhat, Ytest)
    total += err
print('迭代次数为50时，theta_pocket情况下的测试集错误率：',total/2000)
```

    迭代次数为50时，theta_pocket情况下的测试集错误率： 0.132035
Q19：50次更新情况下，最后一次theta作为参数时测试集的错误率

```python
# Q19
total = 0
for i in range(2000):
    theta = np.zeros((row, 1))
    randpos = np.random.permutation(col)
    Xrnd = X[randpos, :]
    Yrnd = Y[randpos, 0:1]
    theta, thetabad = pocket(Xrnd, Yrnd, theta, 50)
    yhat = np.sign(Xtest.dot(thetabad))
    yhat[np.where(yhat == 0)] = -1
    err = mistake(yhat, Ytest)
    total += err
print('迭代次数为50时，theta_50情况下的测试集错误率：',total/2000)
```

    迭代次数为50时，theta_50情况下的测试集错误率： 0.354342

Q20：100次更新情况下的测试集错误率

```python
# Q20
total = 0
for i in range(2000):
    theta = np.zeros((row, 1))
    randpos = np.random.permutation(col)
    Xrnd = X[randpos, :]
    Yrnd = Y[randpos, 0:1]
    theta, thetabad = pocket(Xrnd, Yrnd, theta, 100)
    yhat = np.sign(Xtest.dot(theta))
    yhat[np.where(yhat == 0)] = -1
    err = mistake(yhat, Ytest)
    total += err
print('迭代次数为100时，theta_pocket情况下的测试集错误率：',total/2000)
```

    迭代次数为100时，theta_pocket情况下的测试集错误率： 0.11616

