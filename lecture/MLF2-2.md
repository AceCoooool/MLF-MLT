---
title: 机器学习基石作业2：part2
date: 2017-02-08 14:12:28
tags: MLF&MLT
categories: ML
---

机器学习基石课后作业2-2：对应题目16～题目20
<!-- more -->

## 机器学习基石作业2

## 问题5

Q16~Q20：主要考察“一刀切”式的“决策树桩”算法。以下给出单维和多维情况下的算法的“口语化”说明。其中单维对应的式子：
$$
h_{s,\theta}(x)=s\cdot sign(x-\theta)
$$
多维情况对应的式子：
$$
h_{s,i,\theta}=s\cdot sign(x_i-\theta)
$$

### 算法说明

![单维情况](MLF2-2/Q6.png)

单维树桩算法

>假定初始数据为$\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$
>① 预先设定N个阈值$\theta$（先对数据的$x$进行排序，将$\theta$设定为其间隙值，且取一个最小数左边的值）
>② 计算每一个阈值$\theta$和$s=+1,-1$对应的$E_{in}$，找出其中对应最小$E_{in}$的$\theta,\ s$
>返回$\theta,\ s,\ minE_{in}$

其中①中可以采用其他的策略来实现，但具体方式是相近的。

多维树桩算法

>假定初始数据为$\{(x^{(1)},y^{(1)},(x^{(2)},y^{(2)},...,(x^{(N)},y^{(N)}\}$，其中$x^{(i)}\in\mathbb{R}^d$
>①For i=1,2,...,d:
>​	 寻找维度i情况下的$\theta,\ s,\ minE_{in}$（通过单维树桩的方式求得）
>②寻找上述$d$个不同$minE_{in}$中最小的那个，以及对应的$\theta,\ s$（如果存在两个$minE_{in}$相同则任意取一个）
>返回$\theta,\ s,\ minE_{in}$

### 算法实现

```python
# 单维度决策树桩算法
def decision_stump(X, Y):
    theta = np.sort(X)
    num = len(theta)
    Xtemp = np.tile(X, (num, 1))
    ttemp = np.tile(np.reshape(theta, (num, 1)), (1, num))
    ypred = np.sign(Xtemp - ttemp)
    ypred[ypred == 0] = -1
    err = np.sum(ypred != Y, axis=1)
    if np.min(err) <= num-np.max(err):
        return 1, theta[np.argmin(err)], np.min(err)/num
    else:
        return -1, theta[np.argmax(err)], (num-np.max(err))/num
```
```python
# 多维度决策树桩算法
def decision_stump_multi(X, Y):
    row, col = X.shape
    err = np.zeros((col,)); s = np.zeros((col,)); theta = np.zeros((col,))
    for i in range(col):
        s[i], theta[i], err[i] = decision_stump(X[:, i], Y[:, 0])
    pos = np.argmin(err)
    return pos, s[pos], theta[pos], err[pos]
```
### 具体问题

涉及到自己生成数据问题，生成的数据满足下述两个条件：
(a). $x$产生于$[-1,+1]$上的均匀分布
(b). $y=f(x)+noise$，其中$f(x)=sign(x)$，noise则为有$20\%$的概率翻转$f(x)$的结果

生成数据函数

```python
# 生成数据函数
def generateData():
    x = np.random.uniform(-1, 1, 20)
    y = np.sign(x)
    y[y == 0] = -1
    prop = np.random.uniform(0, 1, 20)
    y[prop >= 0.8] *= -1
    return x, y
```
Q16：对于任意一个决策树桩函数$h_{s,\theta}\ \ \theta\in[-1,+1]$，其对应的$E_{out}(h_{s,\theta})$为以下哪一种函数？
a. $0.3+0.5s(1-|\theta|)$	b. $0.3+0.5s(|\theta|-1)$	c. $0.5+0.3s(|\theta|-1)$	d.$0.5+0.3s(1-|\theta|)$

A16：为简便起见，假设$s=1,\theta\gt0$，此时$h$预测情况：$[\theta,1]\to +1$，$[-1,\theta]\to-1$，$f$真实情况：$(p=0.8)[-1,0]\to-1$，$(p=0.2)[-1,0]\to+1$，$(p=0.8)[0,1]\to+1$，$(p=0.2)[0,1]\to-1$。从而可见错误出现在区间$[0,\theta]$错误概率为$0.8$，其他区域错误概率为$0.2$。因此$E_{out}=(0.2(2-\theta)+0.8\theta)/2=0.2+0.3\theta$，其他三种情况类似分析，最终可得答案为c

Q17&Q18：根据规则随机生成20组数据，运行5,000次，求平均$E_{in}$和平均$E_{out}$（其中$E_{out}$由Q16中的答案来求解）？

```python
# Q17和Q18
totalin = 0; totalout = 0
for i in range(5000):
    X, Y = generateData()
    theta = np.sort(X)
    s, theta, errin = decision_stump(X, Y)
    errout = 0.5+0.3*s*(math.fabs(theta)-1)
    totalin += errin
    totalout += errout
print('训练集平均误差: ', totalin/5000)
print('测试集平均误差: ', totalout/5000)
```

    训练集平均误差:  0.17111
    测试集平均误差:  0.2613195070168455
**Q19&Q20**导入数据集函数

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
Q19&Q20：求多维决策树桩在训练集和测试集上的误差$E_{in}$和$E_{out}$？

```python
# Q19和Q20
X, Y = loadData('hw2_train.dat')
Xtest, Ytest = loadData('hw2_test.dat')
pos, s, theta, err = decision_stump_multi(X, Y)
print('训练集误差: ', err)
ypred = s*np.sign(Xtest[:, pos]-theta)
ypred[ypred == 0] = -1
row, col = Ytest.shape
errout = np.sum(ypred != Ytest.reshape(row,))/len(ypred)
print('测试集误差: ', errout)
```

    训练集误差:  0.25
    测试集误差:  0.355
