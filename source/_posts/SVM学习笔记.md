---
title: SVM学习笔记
date: 2020-04-23 15:57:32
tags: [机器学习]
mathjax: true
---
# 线性SVM详细推导
首先，一个点$p \in \mathbb{R}^d$到超平面$w^Tx+b=0$的距离可以表示为$\frac{1}{||w||}|w^Tp + b|$。

对于一个两类别数据集$X\in\mathbb{R}^d, Y\in\{0, 1\}$，定义间隔$\gamma=2\min_i\frac{1}{||w||}|w^Tx_i + b|$

线性支持向量机的目标即找到一组适合的参数$(w, b)$使得

$$
\max_{w,b}\gamma = \max_{w,b} 2\min_i\frac{1}{||w||}|w^Tx_i + b|\\
s.t. \ y_i(w^Tx_i + b) > 0, i = 1,2,...,m
$$

若一组$(w^*, b^*)$是支持向量机的一个解，那么对于$\lambda > 0$，$(\lambda w^*, \lambda b^*)$也是该优化问题的一个解，因为间隔不会变化。

所以这里添加一个约束条件，让$\min_i |w^T x_i + b| = 1$，因此支持向量机的优化目标可以进一步化成如下，即支持向量机的基本型：

$$
\max_{w,b} 2\min_i\frac{1}{||w||}|w^Tx_i + b| \\
= \max_{w,b} \frac{2}{||w||}\\
= \min_{w,b} \frac{1}{2}w^T w\\
s.t. \ y_i(w^Tx_i + b) \ge 1, i = 1,2,...,m
$$

可以看出，$w^T w$是一个正定二次型，如此一来，支持向量机的可以看做一个凸二次优化问题：

$$
\min_{u} \frac{1}{2} u^T Q u + t^T u \\
s.t. \ c_i^T u \ge d_i, i = 1, 2,...,m
$$

其中$u=\begin{bmatrix}w\\b\end{bmatrix}$, $Q=\begin{bmatrix}I &\mathbf{0}\\\mathbf{0}&0\end{bmatrix}$, $t=\mathbf{0}$, $c_i = y_i\begin{bmatrix}x_i\\ 1\end{bmatrix}$, $d_i = 1$

也可以运用拉格朗日法来求解支持向量机，定义其拉格朗日函数$\mathcal{L}(w,b,\alpha) = \frac{1}{2}w^T w + \sum_{i=1}^{m} \alpha_i(1 - y_i(w^T x_i + b))$
原问题可以表示为
$$
\min_{w,b} \max_\alpha \mathcal{L}(w, b, \alpha)\\
s.t. \ \alpha_i \ge 0, i=1,2,...,m
$$

其对偶问题可以表示为:

$$
\max_\alpha \min_{w,b} \mathcal{L}(w, b, \alpha)\\
s.t. \ \alpha_i \ge 0, i=1,2,...,m
$$

其KKT条件表示为:

$$
\begin{aligned}
&\triangledown_wL(w,b,\alpha) = 0 \\
&\triangledown_bL(w,b,\alpha) = 0 \\
&1 - y_i(w^T x_i + b) \le 0 \\
&\alpha_i \ge 0 \\
&\alpha_i(1 - y_i(w^Tx_i + b)) = 0    
\end{aligned}
$$

这里对支持向量机的对偶问题进行第一步求解:$\min_{w,b} \mathcal{L}(w, b, \alpha)$，直接令一阶导数等于0：
$$
\frac{\partial\mathcal{L}}{\partial w} = 0 \Rightarrow w = \sum_{i=1}^m \alpha_i y_i x_i \\
\frac{\partial\mathcal{L}}{\partial b} = 0 \Rightarrow \sum_{i=1}^m \alpha_i y_i = 0
$$

这里可以看出，$w$仅和$\alpha_i > 0$的样本有关，而根据KKT条件，$\alpha_i > 0$的地方必须满足$(1 - y_i(w^Tx_i + b)) = 0$，即这些$x_i$在最大间隔边界上，这样的样本称为支持向量，支持向量机的解仅仅和支持向量有关。

将求得结果代入$\mathcal{L}$：
$$
\begin{aligned}
\mathcal{L} &= \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j + \sum_{i=1}^m \alpha_i - \sum_{i=1}^m \alpha_i y_i(w^Tx_i + b)\\
&= \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j + \sum_{i=1}^m \alpha_i - \sum_{i=1}^m \alpha_i y_iw^Tx_i\\
&= \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j
\end{aligned}
$$

此时，再求解
$$
\max_{\alpha} \mathcal{L}(w, b, \alpha)\\
s.t. \ \alpha_i \ge 0,i=1,2,...,m\\
\ \sum_{i=1}^m \alpha_i y_i = 0
$$
这个问题的求解使用SMO(序列最小优化)算法，大致思路和坐标上升法类似，迭代进行，每次选取两个$\alpha$进行更新，同时更新参数$b$，具体步骤后面有时间再详细学习。

# SVM的核技巧
线性SVM基于一个基本假设：数据在空间$\mathbb{R}^d$中线性可分，但这个假设在实际应用中，基本不满足。

但是存在一个定理：当$d$有限时，一定存在$\hat{d}$，使得样本在空间$\mathbb{R}^{\hat{d}}$中线性可分。

因此我们可以构造一种映射：$x_i \rightarrow \phi(x_i)$，然后在这个映射的空间中使用线性SVM进行分类。

这样最终需要求解的拉格朗日函数可以写成：$\mathcal{L} = \sum_{i=1}^m \alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m \alpha_i \alpha_j y_i y_j \phi(x_i)^T \phi(x_j)$，看起来很简单，只需要换一种计算方式就行，但是这里存在一个问题：$\hat{d}$可能非常大，导致计算困难。

针对上述问题，需要使用核技巧：构造一个计算复杂度为$O(d)$的$k(x_i, x_j)$使得：$k(x_i, x_j)=\phi(x_i)^T \phi(x_j)$，即我们只需要一种快速的计算内积的方式，并不关心其他运算。

核函数的选择：当数据维数$d$超过了样本数量的时候，一般选用线性核，当数据维数$d$较小，而样本量$m$中等时，可以选择RBF核，但是当数据维数$d$较小，而样本量$m$特别大时，不需要选择了，直接使用深度神经网络吧。

核函数的定义需要满足Mercer条件：核函数矩阵必须是半正定的。可以理解为内积大于等于0。

核函数有一些性质：如果$k_1$、$k_2$是核函数，那么下列函数也是核函数：

$c_1k_1(x_i, x_j) + c_2k_2(x_i, x_j), \ c1,c2 > 0$

$k_1(x_i, x_j)k_2(x_i, x_j)$

$f(x_i)k_1(x_i, x_j)f(x_j)$

# 软间隔SVM
数据中不能总是找到线性可分的空间，而且数据存在噪声或者错误标注，这个时候我们如果按照SVM的优化方式，很可能造成过拟合的问题，因此可以允许少量分类错误出现，定义松弛变量$\epsilon_i = \begin{cases}
    0 & y_i(w^T\phi(x_i) + b) \ge 1\\
    1 - y_i(w^T\phi(x_i) + b) & y_i(w^T\phi(x_i) + b) < 1
\end{cases}$，由此定义软间隔支持向量机的基本型：
$$
\min_{w,b,\epsilon} \frac{1}{2}w^T w + C\sum_{i=1}^m \epsilon_i\\
s.t. y_i(w^T\phi(x_i) + b) \ge 1 - \epsilon_i, i = 1, 2,...,m\\
\epsilon_i \ge 0, i=1,2,...,m
$$
其中$C$是一个可调节参数，用于调节错误分类的惩罚。

软间隔支持向量机的求解方式和支持向量机类似，不过从一个约束变成了两个约束。

另外$\epsilon_i$也可以表示成$\max(0, 1 - y_i(w^T\phi(x_i) + b))$，因此软间隔支持向量机的基本型的对偶问题可以表示为$\min_{w, b} \frac{1}{m}\sum_{i=1}^m \max(0, 1 - y_i(w^T\phi(x_i) + b)) + \frac{1}{2mC}w^Tw$，其中第一项称为经验风险，度量模型对数据的拟合程度，第二项称为结构风险，度量模型的复杂程度，也可以称为正则化项。因此还衍生出一种损失函数：hinge loss：$\mathbb{l}(s) = max(0, 1-s)$。
