---
title: 机器学习技法作业2：part1
date: 2017-02-17 10:07:52
tags: MLF&MLT
categories: ML
---

机器学习技法课后作业2-1：对应题目1～题目11
<!-- more -->

## 机器学习技法作业2

### 问题1

Q1：“概率型”SVM主要优化的表达式如下：
$$
min_{A,B}F(A,B)=\frac{1}{N}\sum_{n=1}^Nln(1+exp(-y_n(A\cdot(w_{svm}^T\phi(x_n)+b_{svm})+B)))
$$
采用梯度下降法来最小化$F(A,B)$时，我们需要计算一阶导数。令$z_n=w_{svm}^T\phi(x_n)+b_{svm}$，以及$p_n=\theta(-y_n(Az_n+B))$，其中$\theta(s)=exp(s)/(1+exp(s))$为常规的logistic函数。则对应的导数$\nabla F(A,B)$为多少？

A1：先进行简单的变量替换$F(A,B)=\frac{1}{N}\sum ln(k)$，在进行一阶偏导数：
$$
\frac{\partial F}{\partial A}=\frac{1}{N}\sum \frac{1}{k}\cdot\frac{\partial k}{\partial A}=\frac{1}{N}\sum_{n=1}^N-y_np_nz_n\\
\frac{\partial F}{\partial B}=\frac{1}{N}\sum \frac{1}{k}\cdot\frac{\partial k}{\partial B}=\frac{1}{N}\sum_{n=1}^N-y_np_n
$$
从而可知最终结果为：$\nabla F(A,B)=\frac{1}{N}\sum_{n=1}^N[-y_np_nz_n,-y_np_n]^T$

### 问题2

Q2：当采用牛顿法来最小化$F(A,B)$时，我们需要计算$-(H(F))^{-1}\nabla F$，其中$H(F)$为$F$的Hessian矩阵，求其具体形式

A2：Hessian矩阵具体可以通过$H(F)=[\frac{\partial^2 F}{\partial A^2},\frac{\partial^2 F}{\partial A\partial B};\frac{\partial^2 F}{\partial A\partial B},\frac{\partial^2 F}{\partial B^2}]$来获得，就是普通的求导，注意细节便可，最终结果如下：
$$
H(F)=\frac{1}{N}\sum_{n=1}^N\begin{bmatrix}z_n^2p_n(1-p_n)&z_np_n(1-p_n)\\z_np_n(1-p_n)&p_n(1-p_n)\end{bmatrix}
$$

### 问题3

Q3：对于$d$维的$N$个输入数据，其在kernel ridge regression中的求逆的矩阵的维数是多少？

A3：在kernel ridge regression中，其参数$\beta$的求解：$\beta=(\lambda I+K)^{-1}Y$，从而可知其对应的求逆矩阵的维数与$K$的大小相等，而$K\to N\times N$，所以选择$N\times N$

### 问题4

Q4：常规(课堂上所学习)的SVR模型通常可以转化为下述优化问题：
$$
(P_1)\quad min_{b,w,\xi^{\lor},\xi^{\land}}\frac{1}{2}w^Tw+C\sum_{n=1}^N(\xi_n^{\lor}+\xi_n^{\land})\\
s.t.\quad -\epsilon-\xi_n^{\lor}\le y_n-w^T\phi(x_n)-b\le \epsilon+\xi_n^{\land}\\
\xi_n^{\lor}\ge 0,\xi_n^{\land}\ge0
$$
大多数情况下，采用线性惩罚的SVR。但另一种$l_2$惩罚的SVR也同样非常常见，含二次惩罚的SVR可以转换为下述优化问题：
$$
(P_2)\quad min_{b,w,\xi^{\lor},\xi^{\land}}\frac{1}{2}w^Tw+C\sum_{n=1}^N((\xi_n^{\lor})^2+(\xi_n^{\land})^2)\\
s.t.\quad -\epsilon-\xi_n^{\lor}\le y_n-w^T\phi(x_n)-b\le \epsilon+\xi_n^{\land}
$$
则与之对应的“无约束”形式的$(P_2)$的形式为？

A4：联系$(P_1)$情况，其对应的“无约束”形式为：$min_{b,w}\frac{1}{2}w^Tw+C\sum_{n=1}^N max(0,|w^Tz_n+b-y_n|-\epsilon)$，即给“越界”情况加上线性惩罚。联系到$(P_2)$，自然而然的可以理解为，给“越界”行为加上二次方惩罚，从而不难知其“无约束”形式为：$min_{b,w}\frac{1}{2}w^Tw+C\sum_{n=1}^N max(0,|w^Tz_n+b-y_n|-\epsilon)^2$

### 问题5

Q5：根据**表示定理(representer theorem)**可知，任意以$L_2$正则项作为惩罚项的线性模型的最优解均可表示为：$w_{\star}=\sum_{n=1}^N\beta_nz_n$。从而将其代入Q4获得的$(P_2)$中将其化为对应的对偶形式。其中用$K(x_n,x_m)=(\phi(x_n))^T(\phi(x_m))$来表示，且令$s_n=\sum_{m=1}^N \beta_mK(x_n,x_m)+b$，不难发现$F(b,\beta)$可以对$\beta$求导。求其导数$\frac{\partial F}{\partial\beta_m}$

A5：首先根据A4的结果给出$(P_2)$的对偶形式，如下所示：
$$
min_{b,\beta}\frac{1}{2}\sum_{n=1}^N\sum_{m=1}^N\beta_n\beta_mK(x_n,x_m)+C\sum_{n=1}^N max(0,|\sum_{k=1}^N\beta_nK(x_k,x_n)+b-y_n|-\epsilon)^2
$$
直接对上式对$\beta_m$求导可得：
$$
\sum_{n=1}^N\beta_nK(x_n,x_m)-2C\sum_{n=1}^N [|y_n-s_n|\ge\epsilon](|y_n-s_n|-\epsilon)sign(y_n-s_n)K(x_n,x_m)
$$
其中需要注意: 当$|y_n-s_n|\gt\epsilon$时，后一项才不为$0$，此外，后一项求导时，注意正负号。

### 问题6

Q6：考虑$T+1$个假设函数$g_0,g_1,...,g_T$，且令$g_0(x)=0,\forall x$。假设有一个潜在的测试集$\{(\hat{x}_m,\hat{y}_m)\}_{m=1}^M$，这个测试集中$\hat{x}_m$你已知，但$\hat{y}_m$你不知道。但是，假设你知道这个测试集对于每一个假设函数的平方误差为$E_{test}(g_t)=\frac{1}{M}\sum_{m=1}^M(g_t(\hat{x}_m)-\hat{y}_m)^2=e_t$，并假设$s_t=\frac{1}{M}\sum_{m=1}^M(g_t(\hat{x}_m))^2=s_t$。则$\sum_{m=1}^Mg_t(\hat{x}_m)\hat{y}_m$可以如何表示？

A6：通过对$E_{test}$展开不难发现：
$$
e_t=\frac{1}{M}\sum_{m=1}^M(g_t(\hat{x}_m)^2-2g_t(\hat{x}_m)\hat{y}_m+\hat{y}_m^2)=s_t-2\frac{1}{M}\sum_{m=1}^Mg_t(\hat{x}_m)\hat{y}_m+e_0\\
\sum_{m=1}^Mg_t(\hat{x}_m)\hat{y}_m=(s_t+e_0-e_t)M/2
$$
"虽然给出了答案，但隐藏在这个背后的核心思想还不是很了解？---是想求出cross-entropy？"

### 问题7

Q7：考虑目标函数$f(x)=x^2:[0,1]\to R$，且输入为$[0,1]$上的均匀分布。假设训练集仅为两个样本，且不存在噪声，采用基于均方损失的线性回归函数$h(x)=w_1x+w_0$进行拟合。求所有假设函数集的数学期望？

A7：由于训练样本是“随机产生”的，且来自均匀分布。假设两个训练样本为$(x_1,x_1^2),(x_2,x_2^2)$，则其对应的损失函数为$L=(w_1x_1+w_0-x_1^2)^2+(w_1x_2+w_0-x_2^2)^2$，对其求导求最优解可得：
$$
\frac{\partial L}{\partial w_1}=2(w_1x_1+w_0-x_1^2)x_1+2(w_1x_2+w_0-x_2^2)x_2=0\\
\frac{\partial L}{\partial w_2}=2(w_1x_1+w_0-x_1^2)+2(w_1x_2+w_0-x_2^2)=0
$$
联立两式可以求解的：$w_1=x_1+x_2,w_0=-x_1x_2$，从而最优函数$g=(x_1+x_2)x-x_1x_2$。根据$x_1,x_2$的选取的所有可能，从而可以获得：
$$
\bar{g}=lim_{T\to\infty}\frac{1}{T}\sum g_t=E(x_1+x_2)x-E(x_1x_2)=x-\frac{1}{4}
$$

### 问题8

Q8：假设AdaBoost中采用的是线性回归函数(用于分类)。则我们可以将问题转化为下述带权值的优化问题：
$$
min_{w}E_{in}^u(w)=\frac{1}{N}\sum_{n=1}^Nu_n(y_n-w^Tx_n)^2
$$
上述的优化问题其实可以视为采用“变形”数据集$\{(\bar{x}_n,\bar{y}_n)\}_{n=1}^N$的普通线性回归常规$E_{in}$的形式，则其“变形”数据$(\bar{x}_n,\bar{y}_n)$与原数据集的关系为？

A8：显然直接将$u_n$放进平方项里面便可获得答案：$(\sqrt{u_n}x_n,\sqrt{u_n}y_n)$

### 问题9

Q9：考虑将AdaBoost算法运用到一个$99\%$数据均为+1的分类问题的样本集上。正是因为有这么多+1的样本，获得第一个最佳假设函数为$g_1(x)=+1$。令$u_{+}^{(2)},u_{-}^{(2)}$分别为第二次迭代时正负样本前的参数，则对应的$u_{+}^{(2)}/u_{-}^{(2)}$的结果为多少？

A9：根据Adaboost的“参数更新规则”，可知：
$$
\frac{u_{+}^{(2)}}{u_{-}^{(2)}}=\frac{\epsilon}{1-\epsilon}=\frac{1}{99}
$$

### 问题10

Q10：在非均匀投票构成的“集成学习”中，存在一个权值向量与下述每次获得的最佳假设函数集进行相乘：
$$
\phi(x)=(g_1(x),g_2(x),...,g_T(x))
$$
在学习kernel模型时，kernel可以视为简单的内积运算：$\phi(x)^T\phi(x^\prime)$。在这个问题中，将这两个主题在决策树桩问题中融合起来。

假设输入变量$x$每一维只取$[L,R]$上的整数，定义下述决策树桩：
$$
g_{s,i,\theta}(x)=s\cdot sign(x_i-\theta)\\
where\quad i\in\{1,2,...,d\},d为x的维数\\
s\in\{-1,+1\},\theta\in\mathbb{R},sign(0)=+1
$$
如果两个决策树桩在任意$x\in\mathcal{X}$上均有$g(x)=\hat{g}(x)$，则认为这两个决策树桩相同。下述哪些表述是正确的？
a. 决策树桩的数量与$\mathcal{X}$的大小有关
b. $g_{s,i,\theta}$和$g_{s,i,celling(\theta)}$相等。其中$celling(\theta)$是指$\ge\theta$的最小整数
c. $\mathcal{X}$的大小为无穷
d. $d=2,L=1,R=6$时，有24种不同的决策树桩

A10：首先需指出可以将$[L,R]$之间按整数换分存在$R-L$段，且不同维度情况是互不干扰的，所以有$2d(R-L)$种情况，又因为还有全正，全负的两种情况，而这两种情况对全部维度是等价的(根据上述定义的等价性可知)，从而总共情况有$total=2d(R-L)+2$种情况。可以直到$a,c,d$均错误。
$g_{s,i,\theta}$和$g_{s,i,celling(\theta)}$对应的决策树桩对于全部$x$均是等价的，因此是相等的。

### 问题11

Q11：Q10的延伸，假设$\mathcal{G}=\{\mathcal{X}上全部可能的决策树桩\}$，并通过下标将其全部罗列出来如下：
$$
\phi_{ds}(x)=(g_1(x),g_2(x),...,g_t(x),...,g_{|\mathcal{G}|}(x))
$$
则：$K_{ds}(x,x^\prime)=\phi_{ds}(x)^T\phi_{ds}(x^\prime)$的一种等价表示方式为什么？其中$||v||_1$表示一阶范数。

A11：首先可以给出$K$的具体表达式如下所示：
$$
K_{ds}=sign(x_{i1}-\theta_1)sign(x_{i1}^\prime-\theta_1)+sign(x_{i2}-\theta_2)sign(x_{i2}^\prime-\theta_2)+...+sign(x_{i\mathcal{G}}-\theta_{\mathcal{G}})sign(x_{i\mathcal{G}}^\prime-\theta_\mathcal{G})
$$
以上囊括了所有的$s,i,\theta$的可能，因此总共包含$2d(R-L)+2$项，现在进一步查看有多少项为$+1$，有多少项为$-1$。查看下述这种情况：
![](MLT2-1/pic1.png)
对于上述含有$a$个整数的情况，总共有$2(a+1)$种假设函数使得其分类结果乘积为$-1$。因此对于给定的$x,x^\prime$（$x,x^\prime$的每一维度上均为整数），总共包含的分类结果乘积为$-1$的数目为：$2||x-x^\prime||_1$，所以分类结果乘积为$+1$的数目为：$2d(R-L)+2-2||x-x^\prime||_1$。所以最终答案为+1项数目减去-1项数目：$2d(R-L)+2-4||x-x^\prime||_1$

