---
title: 机器学习基石作业3：part2
date: 2017-02-09 21:56:45
tags: MLF&MLT
categories: ML
---

机器学习基石课后作业3-2：对应题目13～题目20
<!-- more -->

## 机器学习基石作业3

## 问题5

Q13~Q15：主要关于线性回归问题和特征转换。

数据产生：数据集大小$N=1000$，且$\mathcal{X}=[-1,1]\times[-1,1]$，每个数据的$\mathbb{x}$均等概率的从$\mathcal{X}$中提取。而对应的$y$则根据$f(x_1,x_2)=sign(x_1^2+x_2^2-0.6)$来确定，且对数据集中的$10\%$的数据的$y$进行反转。

先对线性回归算法进行简单的说明

### 算法说明

函数集：$y=w^T\mathbb{x}$

损失函数：$E_{in}(w)=\frac{1}{N}\sum_{n=1}^N(w^T\mathbb{x}_n-y_n)^2$

梯度：$\nabla E_{in}(w)=\frac{2}{N}(X^TXw-X^T\mathbb{y})$

“目的”：寻找$w$使得损失函数最小

Linear Regression

>①获得数据$(\mathbb{x}_1,y_1),...,(\mathbb{x}_N,y_N)$
>②采用闭式解公式求出最佳$w$：$w_{lin}=(X^TX)^{-1}X^T\mathbb{y}$
>③返回$w_{lin}$
>
>如果还有预测过程，直接$\hat{y}=w_{lin}^Tx$

### 算法实现

```python
theta = lin.pinv(X.T.dot(X)).dot(X.T).dot(Y)
```

### 具体问题

数据产生函数：

```python
# 数据生成函数
def generateData(num):
    axeX = np.random.uniform(-1, 1, num)
    axeY = np.random.uniform(-1, 1, num)
    Xtemp = np.c_[axeX, axeY]
    X = np.c_[np.ones((num, 1)), Xtemp]
    Ytemp = np.sign(np.power(axeX, 2)+np.power(axeY, 2)-0.6)
    Ytemp[Ytemp == 0] = -1
    pos = np.random.permutation(num)
    Ytemp[pos[0: round(0.1*num)]] *= -1
    Y = Ytemp.reshape((num, 1))
    return X, Y
```
Q13：不进行特征转换，只采用特征$(1, x_1,x_2)$，利用Linear Regression获得最佳的$w_{lin}$。将其直接运用到分类问题上面（利用$sign(w^Tx)$），在利用$0/1$判据来衡量训练样本误差$E_{in}$。进行1000次实验，取误差的平均。

A13：通过下面的代码来实现：

```python
totalerr = 0
for i in range(1000):
    X, Y = generateData(1000)
    theta = lin.pinv(X.T.dot(X)).dot(X.T).dot(Y)
    ypred = np.sign(X.dot(theta))
    err = np.sum(ypred!=Y)/1000
    totalerr += err
print('Ein: ', totalerr/1000)
```

    Ein:  0.503646

通过上面结果可知，直接利用Linear Regression(利用square error)再运用到分类问题上结果很差！

Q14~Q15：将数据的特征进行转换，转换为$(1,x_1,x_2,x_1x_2,x_1^2,x_2^2)$这6项，再利用Linear Regression获得最佳的$w_{lin}$，求该$w_{lin}$以及将其运用到测试集上的测试误差$E_{out}$（衡量方式与Q13相同）

A14~A15：特征转换函数如下

```python
# 特征转换函数
def transform(X):
    row, col = X.shape
    Xback = np.zeros((row, 6))
    Xback[:, 0:col] = X
    Xback[:, col] = X[:, 1]*X[:, 2]
    Xback[:, col+1] = X[:, 1]**2
    Xback[:, col+2] = X[:, 2]**2
    return Xback
```
问题的具体代码如下：

```python
# Q14
totalerr = 0
for i in range(1000):
    X, Y = generateData(1000)
    Xtran = transform(X)
    theta = lin.pinv(Xtran.T.dot(Xtran)).dot(Xtran.T).dot(Y)
    Xtest, Ytest = generateData(1000)
    Xback = transform(Xtest)
    ypred = np.sign(Xback.dot(theta))
    err = np.sum(ypred!=Ytest)/1000
    totalerr += err
print('theta: ', theta.T)
print('Eout: ', totalerr/1000)
```

    theta:  [[-1.01626639  0.07325707  0.02834912 -0.0155599   1.63387468  1.52477431]]
    Eout:  0.12608
需要指出的是，Q14中给出的选项中最接近的为：
$$
g(x_1,x_2)=sign(-1-0.05x_1+0.08x_2+0.13x_1x_2+1.5x_1^2+1.5x_2^2)
$$

## 问题6

Q16~Q17：关于多类别logistics regression问题。针对K类别分类问题，我们定义输出空间$\mathcal{Y}=\{1,2,...,K\}$，MLR的函数集可以视为由一系列(K个)权值向量$(w_1,...,w_K)$构成，其中每个权值向量均为$d+1$维。每种假设函数可以表示为：
$$
h_y(x)=\frac{exp(w^T_y\mathbb{x})}{\sum_{i=1}^Kexp(w_i^T\mathbb{x})}
$$
且可以用来近似潜在的目标分布函数$P(y|\mathbb{x})$。MLR的“目标”就是从假设函数集中寻找使得似然函数最大的额假设函数。

Q16：类似Lec10中最小化$-log(likelihood)$一样，推导$E_{in}(w_1,...,w_K)$

A16：采用同样的处理方式
$$
max\ \frac{1}{N}\prod_{i=1}^Nh_y(\mathbb{x})\to min\ -\frac{1}{N}\sum_{i=1}^Nlog(h_y(\mathbb{x}))
$$
将MLR的假设函数代入上式并化简可得：
$$
\frac{1}{N}\sum_{n=1}^N\big(ln(\sum_{i=1}^Kexp(w_i^T\mathbb{x}_n))-w^T_{y_n}\mathbb{x}_n\big)
$$
Q17：针对上述的$E_{in}$，它的一阶导数$\nabla E_{in}$可以表示为$\big(\frac{\partial E_{in}}{\partial w_1},\frac{\partial E_{in}}{\partial w_2,},...,\frac{\partial E_{in}}{\partial w_K}\big)$，求$\frac{\partial E_{in}}{\partial w_i}$。

A17：直接对A16的答案的式子进行求导，就可以得到下式：
$$
\frac{1}{N}\sum_{n=1}^N\big((h_i(\mathbb{x}_n)-[y_n=i]\mathbb{x}_n\big)
$$

## 问题7

Q18~Q20：关于logistic regression实现的问题

### 算法说明

函数集：$s=\sum_{i=0}^dw_ix_i$，$h(\mathbb{x})=\theta(s)=\frac{1}{1+e^{-s}}$

损失函数：$E_{in}(w)=\frac{1}{N}\sum_{i=1}^Nln(1+exp(-y_nw^T\mathbb{x}_n))$

梯度：$\nabla E_{in}=\frac{1}{N}\sum_{i=1}^N\theta\big(-y_nw^T\mathbb{x}_n\big)(-y_n\mathbb{x}_n)$

“目的”：寻找一个最佳假设函数使得损失函数最小

（注：$h(\mathbb{x})$来近似$P(y|\mathbb{x})$上述的损失函数通过cross-entropy可推导出来）

Logistic Regression：

>初始化$w$
>For t=0,1,...
>​	① 计算$\nabla E_{in}(w)$
>​	② 更新参数：$w\gets w-\eta\nabla E_{in}(w)$
>返回$w$

（上述$\eta$可以视为一个超参数，可以通过cross-validation来确定）

### 算法实现

```python
# sigmoid函数
def sigmoid(z):
    zback = 1/(1+np.exp(-1*z))
    return zback
```
```python
# Logistic Regression
def logisticReg(X, Y, eta, numiter, flag=0):
    row, col = X.shape
    theta = np.zeros((col, 1))
    num = 0
    for i in range(numiter):
        if flag == 0:
            derr = (-1*X*Y).T.dot(sigmoid(-1*X.dot(theta)*Y))/row
        else:
            if num >= row:
                num = 0
            derr = -Y[num, 0]*X[num: num+1, :].T*sigmoid(-1*X[num, :].dot(theta)[0]*Y[num, 0])
            num += 1
        theta -= eta*derr
    return theta
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
错误率计算模块

```python
# 误差计算函数
def mistake(X, Y, theta):
    yhat = X.dot(theta)
    yhat[yhat > 0] = 1
    yhat[yhat <= 0] = -1
    err = np.sum(yhat != Y)/len(Y)
    return err
```
Q18：针对$\eta=0.001,\ T=2000$的情况，采用梯度下降法获得$w$后，在测试集上的错误率是多少？（利用0/1判据）

A18：

```python
# Q18
eta = 0.001; T = 2000; flag = 0
theta = logisticReg(X, Y, eta, T, flag)
errin = mistake(X, Y, theta)
errout = mistake(Xtest, Ytest, theta)
print('Ein = ', errin,'Eout = ', errout)
```

    Ein =  0.466 Eout =  0.475

Q19：针对$\eta=0.01,\ T=2000$的情况，采用梯度下降法获得$w$后，在测试集上的错误率是多少？（利用0/1判据）

A19：

```python
# Q19
eta = 0.01; T = 2000; flag = 0
theta = logisticReg(X, Y, eta, T, flag)
errin = mistake(X, Y, theta)
errout = mistake(Xtest, Ytest, theta)
print('Ein = ', errin,'Eout = ', errout)
```

    Ein =  0.197 Eout =  0.22

Q20：针对$\eta=0.001,\ T=2000$的情况，采用随机梯度下降法(此处采用按顺序每次选择元素，更通常的做法是随机选择元素)获得$w$后，在测试集上的错误率是多少？（利用0/1判据）

A20：

```python
# Q20
eta = 0.001; T = 2000; flag = 1
theta = logisticReg(X, Y, eta, T, flag)
errin = mistake(X, Y, theta)
errout = mistake(Xtest, Ytest, theta)
print('Ein = ', errin,'Eout = ', errout)
```

    Ein =  0.464 Eout =  0.473

