---
title: 机器学习技法作业3：part2
date: 2017-02-23 11:35:47
tags: MLF&MLT
categories: ML
---

机器学习技法课后作业3-1：对应题目13～题目20
<!-- more -->

## 机器学习技法作业3

### 问题5

Q13~Q15：关于决策树的问题。

基于课堂所介绍的C&RT算法：基于Gini系数，无剪枝。来实现并对所给的数据集进行实验。

```python
# 为了实现决策树，需建立树结点的类
# 定义树结点
class Node:
    def __init__(self, theta, index, value=None):
        self.theta = theta     # 划分的阈值
        self.index = index     # 选用的维度
        self.value = value     # 根节点的值
        self.leftNode = None
        self.rightNode = None
```
```python
# 定义Gini系数---作为每个子集“好坏”的衡量标准
def gini(Y):
    l = Y.shape[0]
    if l == 0:
        return 1
    return 1-(np.sum(Y==1)/l)**2-(np.sum(Y==-1)/l)**2
```
```python
# 为了便于实现，找出每一维度下的最佳划分阈值和对应的branch值 --- 但这样实现代价是运行速度
# 单维情况下的最佳树桩---大于等于为1类
def one_stump(X, Y, thres):
    l = thres.shape[0]
    mini = Y.shape[0]
    for i in range(l):
        Y1 = Y[X<thres[i]]
        Y2 = Y[X>=thres[i]]
        judge = Y1.shape[0]*gini(Y1)+Y2.shape[0]*gini(Y2)
        if mini>judge:
            mini = judge; b = thres[i]
    return mini, b
```
```python
# 定义划分终止的条件
# 终止条件
def stop_cond(X, Y):
    if np.sum(Y!=Y[0])==0 or X.shape[0]==1 or np.sum(X!=X[0, :])==0:
        return True
    return False
```
```python
# 定义完全生长的决策树
def dTree(X, Y):
    if stop_cond(X, Y):
        node = Node(None, None, Y[0])
        return node
    b, index = decision_stump(X, Y)
    pos1 = X[:, index] < b; pos2 = X[:, index] >= b
    leftX = X[pos1, :]; leftY = Y[pos1, 0:1]
    rightX = X[pos2, :]; rightY = Y[pos2, 0:1]
    node = Node(b, index)
    node.leftNode = dTree(leftX, leftY)
    node.rightNode = dTree(rightX, rightY)
    return node
```
```python
# 预测函数---基于决策树对单个样本进行的预测
def predict_one(node, X):
    if node.value is not None:
        return node.value[0]
    thre = node.theta; index = node.index
    if X[index] < thre:
        return predict_one(node.leftNode, X)
    else:
        return predict_one(node.rightNode, X)
```
```python
# 基于决策树的预测结果及其错误率衡量函数
def err_fun(X, Y, node):
    row, col = X.shape
    Yhat = np.zeros(Y.shape)
    for i in range(row):
        Yhat[i] = predict_one(node, X[i, :])
    return Yhat, np.sum(Yhat!=Y)/row
```
Q13：获得的决策树$G$含有多少内部结点(不包括叶子结点，即所有包含阈值分割的结点)？

A13：需要定义一个搜索树有几个结点的函数

```python
# Q13
# 定义一个搜索树有多少结点的函数---叶子结点不计入
def internal_node(node):
    if node == None:
        return 0
    if node.leftNode == None and node.rightNode == None:
        return 0
    l = 0; r = 0
    if node.leftNode != None:
        l = internal_node(node.leftNode)
    if node.rightNode != None:
        r = internal_node(node.rightNode)
    return 1 + l + r

node = dTree(X, Y)
print('完全生长的决策树内部结点数目：', internal_node(node))
```

    完全生长的决策树内部结点数目： 10

Q14和Q15：基于$0/1$误差判据的训练集和验证集的错误率$E_{in},E_{out}$分别是多少？

A14和A15：

```python
# Q14 and Q15
_, ein = err_fun(X, Y, node)
_, eout = err_fun(Xtest, Ytest, node)
print('Ein: ', ein, '\nEout: ', eout)
```

    Ein:  0.0 
    Eout:  0.126

### 问题6

Q16~Q18：关于随机森林算法

采用$N^\prime=N$的bagging策略，并结合上述实现的决策树算法，构造随机森林。实践中进行$T=300$次bagging(即生成300棵树)，并进行100次(为了节约时间，采用50次)实验取平均的$E_{in},E_{out}$

```python
# bagging函数
def bagging(X, Y):
    row, col = X.shape
    pos = np.random.randint(0, row, (row,))
    return X[pos, :], Y[pos, :]
```
```python
# 随机森林算法---没有加入feature的随机选择
def random_forest(X, Y, T):
    nodeArr = []
    for i in range(T):
        Xtemp, Ytemp = bagging(X, Y)
        node = dTree(Xtemp, Ytemp)
        nodeArr.append(node)
    return nodeArr
```
Q16~Q18：分别求30000(此处采用15000)棵决策树误差的平均$E_{in}(g_t)$，以及平均误差$E_{in}(G_{RF}),E_{out}(G_{RF})$

```python
# Q16,Q17,Q18
ein = 0; eout = 0; err = 0
for j in range(50):
    nodeArr = random_forest(X, Y, 300)
    l = len(nodeArr)
    yhat1 = np.zeros((Y.shape[0], l))
    yhat2 = np.zeros((Ytest.shape[0], l))
    for i in range(l):
        yhat1[:, i:i+1], _ = err_fun(X, Y, nodeArr[i])
        yhat2[:, i:i+1], _ = err_fun(Xtest, Ytest, nodeArr[i])
    errg = np.sum(yhat1!=Y, 0)/Y.shape[0]
    Yhat = np.sign(np.sum(yhat1, 1)).reshape(Y.shape)
    Ytesthat = np.sign(np.sum(yhat2, 1)).reshape(Ytest.shape)
    Yhat[Yhat == 0] = 1; Ytesthat[Ytesthat == 0] = 1
    ein += np.sum(Yhat!=Y)/Y.shape[0]
    eout += np.sum(Ytesthat!=Ytest)/Ytest.shape[0]
    err += np.sum(errg)/l
print('Ein(gt)的平均：', err/50)
print('Ein(G): ', ein/50)
print('Eout(G): ', eout/50)
```

    Ein(gt)的平均： 0.0518873333333
    Ein(G):  0.0
    Eout(G):  0.07452

### 问题7

Q19~Q20：基于“超级”剪枝的随机森林算法

将每棵决策树限制为只有一个分支的情况。即相当于变成了一个决策树桩。与问题6中采用同样的bagging策略。重复实验，求对应的平均$E_{in}(G_{RS}),E_{out}(G_{RS})$

```python
# 定义只进行一次划分的决策树（夸张的剪枝）
def dTree_one(X, Y):
    b, index = decision_stump(X, Y)
    pos1 = X[:, index] < b; pos2 = X[:, index] >= b
    node = Node(b, index)
    value1 = 1 if np.sign(np.sum(Y[pos1]))>=0 else -1
    value2 = 1 if np.sign(np.sum(Y[pos2]))>=0 else -1
    node.leftNode = Node(None, None, np.array([value1]))
    node.rightNode = Node(None, None, np.array([value2]))
    return node
```
```python
# 基于剪枝后的随机森林算法
def random_forest_pruned(X, Y, T):
    nodeArr = []
    for i in range(T):
        Xtemp, Ytemp = bagging(X, Y)
        node = dTree_one(Xtemp, Ytemp)
        nodeArr.append(node)
    return nodeArr
```
A19~A20：

```python
# Q19, Q20
ein = 0; eout = 0
for j in range(50):
    nodeArr = random_forest_pruned(X, Y, 300)
    l = len(nodeArr)
    yhat1 = np.zeros((Y.shape[0], l))
    yhat2 = np.zeros((Ytest.shape[0], l))
    for i in range(l):
        yhat1[:, i:i + 1], _ = err_fun(X, Y, nodeArr[i])
        yhat2[:, i:i + 1], _ = err_fun(Xtest, Ytest, nodeArr[i])
    Yhat = np.sign(np.sum(yhat1, 1)).reshape(Y.shape)
    Ytesthat = np.sign(np.sum(yhat2, 1)).reshape(Ytest.shape)
    Yhat[Yhat == 0] = 1;
    Ytesthat[Ytesthat == 0] = 1
    ein += np.sum(Yhat != Y) / Y.shape[0]
    eout += np.sum(Ytesthat != Ytest) / Ytest.shape[0]
print('Ein: ', ein/50)
print('Eout: ', eout/50)
```

    Ein:  0.1106
    Eout:  0.15336

