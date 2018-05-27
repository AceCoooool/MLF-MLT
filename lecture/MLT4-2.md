---
title: 机器学习技法作业4：part2
date: 2017-02-25 15:52:03
tags: MLF&MLT
categories: ML
---

机器学习技法课后作业4-2：对应题目11～题目20
<!-- more -->

## 机器学习技法作业4

### 问题5

Q11~Q14：BP神经网络

根据Lec12p16实现$d-M-1$神经网络算法，采用的细节①神经元采用**tanh**函数(输出层也采用该函数)。②误差函数采用平方损失函数 ③采用随机梯度下降法 ④ 进行$T=50000$次参数更新

- 隐藏层神经元数目为$M$
- $w_{ij}^{(l)}$初始化为$(-r,r)$上的均匀分布
- $\eta$为学习率

```python
# 初始化theta函数
def inittheta(d, M, r):
    theta1 = np.random.uniform(-r, r, (d, M))
    theta2 = np.random.uniform(-r, r, (M+1, 1))
    return theta1, theta2
```
```python
# tanh的导数函数
def dertanh(s):
    return 1-np.tanh(s)**2
```
```python
# 神经网络函数---BP更新参数
def nnetwork(X, Y, M, r, eta, T):
    row, col = X.shape
    theta1, theta2 = inittheta(col, M, r)
    for i in range(T):
        # 前向传播
        randpos = np.random.randint(0, row)
        xone = X[randpos: randpos+1, :]
        yone = Y[randpos]
        s1 = xone.dot(theta1)
        x1 = np.tanh(s1)
        x1 = np.c_[np.ones((1, 1)), x1]
        s2 = x1.dot(theta2)
        x2 = np.tanh(s2)[0][0]
        delta2 = -2*(yone-x2)
        delta1 = delta2*theta2[1:, :].T*dertanh(s1)
        theta2 -= eta*x1.T*delta2
        theta1 -= eta*xone.T.dot(delta1)
    return theta1, theta2
```
```python
# 误差衡量函数
def errfun(X, Y, theta):
    row, col = X.shape
    l = len(theta)
    x = X
    for i in range(l-1):
        x = np.c_[np.ones((row, 1)), np.tanh(x.dot(theta[i]))]
    x2 = np.tanh(x.dot(theta[l-1]))
    Yhat = x2
    Yhat[Yhat>=0] = 1
    Yhat[Yhat<0] = -1
    return np.sum(Yhat != Y)/row
```
Q11：已知$\eta=0.1,r=0.1$，考虑$M\in\{1,6,11,16,21\}$并重复进行500次实验，则哪个$M$对应的平均$E_{out}$最小？

A11：

```python
# Q11
M = [1, 6, 11, 16, 21]
eout = np.zeros((len(M),))
for i in range(500):
    for j in range(len(M)):
        theta1, theta2 = nnetwork(X, Y, M[j], 0.1, 0.1, 50000)
        theta = [theta1, theta2]
        eout[j] += errfun(Xtest, Ytest, theta)
print(eout/500)
```

    M=    		1        6		 11		   16		  21	
    eout= [ 0.307912  0.036136  0.036264  0.03644   0.036336]
Q12：已知$\eta=0.1,M=3$，考虑$r\in\{0,0.1,10,100,1000\}$并重复进行500次实验(此处采用了50次实验)，则哪个$r$对应的平均$E_{out}$最小？

A12：

```python
# Q12
r = [0, 0.1, 10, 100, 1000]
eout = np.zeros((len(r),))
for i in range(50):
    for j in range(len(r)):
        theta1, theta2 = nnetwork(X, Y, 3, r[j], 0.1, 50000)
        theta = [theta1, theta2]
        eout[j] += errfun(Xtest, Ytest, theta)
print(eout / 50)
```

    r=    	 0        0.1		 10		 100	  1000	
    eout= [ 0.49328  0.036    0.15016  0.40392  0.41504]
Q13：已知$r=0.1,M=3$，考虑$\eta\in\{0.001,0.01,0.1,1,10\}$并重复进行500次实验(此处采用了50次实验)，则哪个$\eta$对应的平均$E_{out}$最小？

A13：

```python
# Q13
eta = [0.001, 0.01, 0.1, 1, 10]
eout = np.zeros((len(eta),))
for i in range(50):
    for j in range(len(eta)):
        theta1, theta2 = nnetwork(X, Y, 3, 0.1, eta[j], 50000)
        theta = [theta1, theta2]
        eout[j] += errfun(Xtest, Ytest, theta)
print(eout / 50)
```

    eta=    0.001      0.01	   0.1		 1	      10	
    eout= [ 0.1044   0.03584  0.036    0.3788   0.47104]
Q14：扩展网络，将其变为$d-8-3-1$型的神经网络，其他与之前网络均类似。已知$r=0.1,\eta=0.01$，重复进行500次实验(此处采用了50次实验)，则对应的$E_{out}$的平均为多少？

A14：由于之前实现的算法并不具备扩展性，因此重新建立下述的神经网络函数：

```python
# 多层神经网络
def nnetwork2hidden(X, Y, d1, d2, T):
    row, col = X.shape
    theta1 = np.random.uniform(-0.1, 0.1, (col, d1))
    theta2 = np.random.uniform(-0.1, 0.1, (d1+1, d2))
    theta3 = np.random.uniform(-0.1, 0.1, (d2+1, 1))
    for i in range(T):
        # 前向传播
        randpos = np.random.randint(0, row)
        xone = X[randpos: randpos+1, :]
        yone = Y[randpos]
        s1 = xone.dot(theta1)
        x1 = np.tanh(s1)
        x1 = np.c_[np.ones((1, 1)), x1]
        s2 = x1.dot(theta2)
        x2 = np.tanh(s2)
        x2 = np.c_[np.ones((1, 1)), x2]
        s3 = x2.dot(theta3)
        x3 = np.tanh(s3)[0][0]
        delta3 = -2*(yone-x3)
        delta2 = delta3*theta3[1:, :].T*dertanh(s2)
        delta1 = delta2.dot(theta2[1:, :].T)*dertanh(s1)
        theta3 -= 0.01*x2.T*delta3
        theta2 -= 0.01*x1.T*delta2
        theta1 -= 0.01*xone.T.dot(delta1)
    return theta1, theta2, theta3
```
```python
# Q14
eout = 0
for i in range(50):
    theta1, theta2, theta3 = nnetwork2hidden(X, Y, 8, 3, 50000)
    theta = [theta1, theta2, theta3]
    eout += errfun(Xtest, Ytest, theta)
print(eout/50)
```

    eout = 0.036

### 问题6

Q15~Q18：knn算法

```python
#---------kNN----------------
def kNNeighbor(k, xpred, X, Y):
    xmin = np.sum((xpred - X)**2, 1)
    pos = np.argsort(xmin, 0)
    Ypred = Y[pos[0:k]]
    Ypred = np.sum(Ypred)
    Ypred = 1 if Ypred>=0 else -1
    return Ypred
```
```python
# 预测函数
def predict(Xtest, X, Y, k):
    row, col = Xtest.shape
    Ypred = np.zeros((row, 1))
    for i in range(row):
        Ypred[i] = kNNeighbor(k, Xtest[i, :], X, Y)
    return Ypred
```
Q15~Q16：考虑$k=1$时的情况，求对应的$E_{in},E_{out}$

A15~A16：$E_{in}=0$是显然可见的。

```python
# Q15-Q16
Yhat = predict(Xtest, X, Y, 1)
eout = np.sum(Yhat!=Ytest)/Ytest.shape[0]
print(eout)
```

    eout = 0.344

Q17~Q18：考虑$k=5$的情况，求对应的$E_{in},E_{out}$

```python
# Q17-Q18
Yhat1 = predict(X, X, Y, 5)
Yhat2 = predict(Xtest, X, Y, 5)
ein = np.sum(Yhat1 != Y) / Y.shape[0]
eout = np.sum(Yhat2 != Ytest) / Ytest.shape[0]
print(ein, eout)
```

     ein	 eout
    0.16    0.316

### 问题7

Q19~Q20：kMean实验

此处定义的kMean的误差函数为：$E_{in}=\frac{1}{N}\sum_{n=1}^N\sum_{m=1^M} [[x_n\in S_m]]||x_n-\mu_m||^2$

```python
# -----------kMeans------------
def kMean(k, X):
    row, col = X.shape
    pos = np.random.permutation(row)
    mu = X[pos[0: k], :]
    epsilon = 1e-5; simi = 1
    while simi>epsilon:
        S = np.zeros((row, k))
        for i in range(k):
            S[:, i] = np.sum((X-mu[i, :])**2, 1)
        tempmu = mu.copy()
        pos = np.argmin(S, 1)
        for i in range(k):
            mu[i, :] = np.mean(X[pos == i, :], 0)
        simi = np.sum(tempmu-mu)
    return mu
```


```python
# 误差函数
def errfun(X, mu):
    row, col = X.shape
    k = mu.shape[0]
    err = 0
    S = np.zeros((row, k))
    for i in range(k):
        S[:, i] = np.sum((X - mu[i, :]) ** 2, 1)
    pos = np.argmin(S, 1)
    for i in range(k):
        err += np.sum((X[pos == i, :]-mu[i, :])**2)
    return err/row
```
Q19：令$k=2$，进行100次实验，求$E_{in}$的平均

A19：

```python
# Q19
err = 0
for i in range(100):
    mu = kMean(2, X)
    err += errfun(X, mu)
print(err/100)
```

    ein = 2.71678714378

Q20：令$k=10$，进行100次实验，求$E_{in}$的平均

A20：

```python
# Q20
err = 0
for i in range(100):
    mu = kMean(10, X)
    err += errfun(X, mu)
print(err/100)
```

    ein = 1.79117604501
