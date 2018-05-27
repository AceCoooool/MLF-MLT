---
title: 机器学习技法作业1：part2
date: 2017-02-14 15:15:09
tags: MLF&MLT
categories: ML
---

机器学习技法课后作业1-2：对应题目15～题目20
<!-- more -->

## 机器学习技法作业1

Q15~Q20主要考虑在一个实际数据集上进行实验。该数据集为邮票数字，输入为数字对应图像的密度和对称度，输出为对应的数字。我们主要考虑多分类问题中一对多的情况。
注意事项：在计算$E_{in},E_{out},E_{val}$时，采用$0/1$判据，此外别对数据进行处理（如缩放等）

```python
# Q15~Q20
# 加载数据函数
def loadData(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X = np.c_[np.ones((col, 1)), data[:, 1: row]]
    Y = data[:, 0]
    return X, Y
```
```python
# 误差计算函数
def mistake(yhat, y):
    err = np.sum(yhat != y)/len(y)
    return err
```
```python
# 导入数据
X, Y = loadData('features_train.dat')
Xtest, Ytest = loadData('features_test.dat')
row, col = X.shape
```
### 问题15

Q15：采用线性kernel: $K(x_n,x_m)=x_n^Tx_m$，取$C=0.01$，解决$(0..VS..非0)$的二元分类问题，则其解得的$||w||$的值最接近多少？

A15：以下结果可见，应该选择$0.6$

```python
# Q15
Ytemp = Y.copy()
pos1 = Ytemp == 0; pos2 = Ytemp != 0
Ytemp[pos1] = 1; Ytemp[pos2] = -1
clf = SVC(C=0.01, kernel='linear', shrinking=False)
clf.fit(X, Ytemp)
print('w: ', clf.coef_, '\n |w|: ', np.linalg.norm(clf.coef_))
```

    w:  [[ -1.82145965e-15   5.70727340e-01   2.59535779e-02]] 
     |w|:  0.571317149084
### 问题16

Q16：考虑采用多项式kernel: $K(x_n,x_m)=(1+x_n^Tx_m)^Q$，考虑参数$C=0.01，Q=2$的情况下，各种一对多情况的$E_{in}$

A16：从而可见应该选择$8..vs..not.8$

```python
# Q16~Q17
Ein = np.zeros((10,))
Salpha = np.zeros((10,))
clf = SVC(C=0.01, kernel='poly', degree=2, gamma=1, coef0=1, shrinking=False)
for i in range(10):
    Ytemp = Y.copy()
    pos1 = Ytemp == i; pos2 = Ytemp != i
    Ytemp[pos1] = 1; Ytemp[pos2] = -1
    clf.fit(X, Ytemp)
    Yhat = clf.predict(X)
    Ein[i] = mistake(Ytemp, Yhat)
    Salpha[i] = np.sum(np.abs(clf.dual_coef_))
out = np.c_[Ein,Salpha]
print('\tEin\t\t Sum_alpha')
print(out)
```

    	   Ein		         Sum_alpha
    [[  1.02455082e-01   2.14119479e+01]
     [  1.44013167e-02   3.74000000e+00]
     [  1.00260595e-01   1.46200000e+01]
     [  9.02482513e-02   1.31600000e+01]
     [  8.94253189e-02   1.30400000e+01]
     [  7.62584008e-02   1.11200000e+01]
     [  9.10711837e-02   1.32800000e+01]
     [  8.84652311e-02   1.29000000e+01]
     [  7.43382252e-02   1.08400000e+01]
     [  8.83280757e-02   1.28800000e+01]]

### 问题17

Q17：与Q16相同的情况下，求各自对应的$\sum\alpha_n$

A17：从上述程序可见，应该选择$20.0$

### 问题18

Q18：考虑高斯kernel: $K(x_n,x_m)=exp(-\gamma||x_n-x_m||^2)$。当$\gamma=100$，考虑$0..vs..not.0$问题，分别取$C=[0.001,0.01,0.1,1,10]$这五种情况时，以下对应的几种属性中哪个是随着$C$增大严格递减的？
(a). $\mathcal{Z}$空间中，自由支持向量到超平面的距离
(b). 支持向量的数量
(c). $E_{out}$
(d). $\sum_{n=1}^N\sum_{m=1}^NK(x_n,x_m)$

A18：显然(d)项是一个定值，不变。(a)项将$C$视为惩罚因子，$C$越大，容忍能力越差，从而会导致margin会更小，因此(a)项正确。其他两项看下面结果

```python
# Q18
c = np.array([0.001, 0.01, 0.1, 1, 10])
nsup = np.zeros((len(c),))
eout = np.zeros((len(c),))
Ytemp = Y.copy()
pos1 = Ytemp == 0; pos2 = Ytemp != 0
Ytemp[pos1] = 1; Ytemp[pos2] = -1
Ytesttemp = Ytest.copy()
pos1 = Ytesttemp == 0; pos2 = Ytesttemp != 0
Ytesttemp[pos1] = 1; Ytesttemp[pos2] = -1
for i in range(len(c)):
    clf = SVC(C=c[i], kernel='rbf', gamma=100, shrinking=False)
    clf.fit(X, Ytemp)
    nsup[i] = np.sum(clf.n_support_)
    yhat = clf.predict(Xtest)
    eout[i] = mistake(Ytesttemp, yhat)
out = np.c_[np.c_[c,nsup],eout]
print('\tC\t\t n_suport\t eout')
print(out)
```

    	   C		         n_suport	       eout
    [[  1.00000000e-03   2.39800000e+03   1.78873941e-01]
     [  1.00000000e-02   2.52000000e+03   1.78873941e-01]
     [  1.00000000e-01   2.28500000e+03   1.05132038e-01]
     [  1.00000000e+00   1.77400000e+03   1.03637270e-01]
     [  1.00000000e+01   1.67300000e+03   1.04633782e-01]]

### 问题19

Q19：内容同Q18，当$C=0.1$时，$\gamma=[1,10,100,1000,10000]$中哪个对应最下的$E_{out}$

A19：从下述结果可见，选择$\gamma=10$

```python
# Q19
gamma1 = np.array([1, 10, 100, 1000, 10000])
eout = np.zeros((len(gamma1),))
Ytemp = Y.copy()
pos1 = Ytemp == 0; pos2 = Ytemp != 0
Ytemp[pos1] = 1; Ytemp[pos2] = -1
Ytesttemp = Ytest.copy()
pos1 = Ytesttemp == 0; pos2 = Ytesttemp != 0
Ytesttemp[pos1] = 1; Ytesttemp[pos2] = -1
for i in range(len(gamma1)):
    clf = SVC(C=0.1, kernel='rbf', gamma=gamma1[i], shrinking=False)
    clf.fit(X, Ytemp)
    yhat = clf.predict(Xtest)
    eout[i] = mistake(yhat, Ytesttemp)
out = np.c_[gamma1, eout]
print('\t gamma \t\t eout')
print(out)
```

    	   gamma 		     eout
    [[  1.00000000e+00   1.07125062e-01]
     [  1.00000000e+01   9.91529646e-02]
     [  1.00000000e+02   1.05132038e-01]
     [  1.00000000e+03   1.78873941e-01]
     [  1.00000000e+04   1.78873941e-01]]

### 问题20

Q20：内容同Q18，将数据集中随机选取1000个作为验证集，则对于$C=0.1$的情况下，不同的$\gamma=[1,10,100,1000,10000]$中，哪个对应的$E_{val}$最小（进行100次实验取平均）

A20：从下述结果可见，$\gamma=10$时对应最小

```python
# Q20
evali = np.zeros((len(gamma1),))
Ytemp = Y.copy()
pos1 = Ytemp == 0; pos2 = Ytemp != 0
Ytemp[pos1] = 1; Ytemp[pos2] = -1
for i in range(len(gamma1)):
    for j in range(100):
        pos = np.random.permutation(row)
        Xval = X[pos[0:1000], :]; Yval = Ytemp[pos[0:1000]]
        Xtrain = X[pos[1000:], :]; Ytrain = Ytemp[pos[1000:]]
        clf = SVC(C=0.1, kernel='rbf', gamma=gamma1[i], shrinking=False)
        clf.fit(Xtrain, Ytrain)
        yhat = clf.predict(Xval)
        evali[i] += mistake(yhat, Yval)
out = np.c_[gamma1, evali/100]
print('\t gamma\t\t eval')
print(out)
```

    	   gamma		      eval
    [[  1.00000000e+00   1.05900000e-01]
     [  1.00000000e+01   9.94500000e-02]
     [  1.00000000e+02   1.00830000e-01]
     [  1.00000000e+03   1.64670000e-01]
     [  1.00000000e+04   1.62690000e-01]]