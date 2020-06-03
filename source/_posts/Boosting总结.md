---
title: Boosting总结
date: 2020-05-28 16:56:22
tags: [机器学习]
mathjax: true
---

# Boosting
Boosting是集成学习的一类算法的总称，这类算法的流程可以概括为：首先从初始训练集中训练出一个基学习器，然后根据基学习器的表现对训练样本分布进行调整，使得之前做错的样本在后面受到更多的关注，然后基于调整后的样本分布来训练下一个基学习器，如此逐步增加基学习器，直到基学习器的数目达到了指定值T。

## AdaBoost
AdaBoost是Boosting方法中的一个代表，主要应用在二分类任务上，AdaBoost可以理解为将基学习进行线性组合得到最终的学习器，其中$\alpha_t$是一个正数，表示第$t$个基学习器的权重，如下所示。

$$
H(x) = \sum\limits_{t=1}^T \alpha_t h_t(x)
$$

AdaBoost使用指数损失函数作为优化目标，如下所示，其中$f(x)$表示真实函数，$f(x) \in \{1, -1\}$，优化这个指数损失函数其实相当于优化0/1损失函数（证明略），因为指数损失函数的平滑性质更利于优化过程，因此这里用指数损失函数，其中$E$表示期望，$D$表示原始数据的分布（每个数据出现的概率是$\frac{1}{N}$，$N$为数据个数）。

$$
l(H|D) = E_{x \sim D}[e^{-f(x)H(x)}]
$$

因此AdaBoost的目标可以表示为：
$$
\mathop{\arg\min}\limits_{\alpha, h} E_{x \sim D}[e^{-f(x)(\sum\limits_{t=1}^T \alpha_t h_t(x))}]
$$

但是这个问题的直接求解非常困难，因此这里使用一种**前向分布算法**来解决这个问题，其思路是：每增加一个基学习器，就让损失函数减小一些，这样来逐步逼近最优解。

假设现在已经得到$m$个基学习器组成的集成学习器：$H_m(x)$，现在下一个学习器（$\alpha_{m+1}, h_{m+1}$）的目标可以表示为：
$$
\begin{aligned}
    &\mathop{\arg\min}\limits_{\alpha_{m+1}, h_{m+1}} E_{x \sim D}[e^{-f(x)(H_m(x) + \alpha_{m+1} h_{m+1}(x))}]\\
    &=\mathop{\arg\min}\limits_{\alpha_{m+1}, h_{m+1}} E_{x \sim D}[e^{-f(x)(H_m(x)} \times e^{-f(x) \alpha_{m+1} h_{m+1}(x))}]
\end{aligned}
$$

$H_m(x)$和$\alpha_{m+1}, h_{m+1}$无关，因此可以将$e^{-f(x)(H_m(x)}$看做数据的一种权重，即使用$e^{-f(x)(H_m(x)}$这一项来改变数据的分布，假设这样改变数据权重之后，得到的数据分布为$D_m$，那么可以进一步将目标化简为：

$$
\mathop{\arg\min}\limits_{\alpha_{m+1}, h_{m+1}} E_{x \sim D_m}\ e^{-f(x) \alpha_{m+1} h_{m+1}(x)}
$$

因为$f(x)$和$h_m{x}$的取值要么是1，要么是$-1$，又因为$\alpha_m$为正数，因此这里如果要最小化目标，可以先不考虑$\alpha_m$，而是先最小化：

$$
\mathop{\arg\min}\limits_{h_{m+1}} E_{x \sim D_m}\ e^{-f(x)h_{m+1}(x)}
$$

因此第$m+1$个基学习器的目标就是在数据分布$D_m$上优化指数损失函数：

$$
\begin{aligned}
h^\star_{m+1} &= \mathop{\arg\min}\limits_{h_{m+1}} E_{x \sim D_m}\ e^{-f(x)h_{m+1}(x)}\\
&=\mathop{\arg\min}\limits_{h_{m+1}} \sum\limits_{i=1}^N w_{mi}I\{f(x_i) \ne h_{m+1}(x_i)\}
\end{aligned}
$$

用$w_{mi} = \frac{e^{-f(x_i)(H_m(x_i)}}{\sum\limits_{i=1}^N e^{-f(x_i)(H_m(x_i)}}$来表示数据$x_i$的权重，这里多出来了一个用于权重归一化的分母，但是不影响$\alpha_{m+1}，h_{m+1}$的训练，只是后面推导会方便些。

第$m+1$个基学习器$h^\star_{m+1}$得到之后，再来确定其权重$\alpha_{m+1}$，其目标是：

$$
\begin{aligned}
    &\mathop{\arg\min}\limits_{\alpha_{m+1}} E_{x \sim D_m}\ e^{-f(x) \alpha_{m+1} h^\star_{m+1}(x)}\\
    &=\mathop{\arg\min}\limits_{\alpha_{m+1}} \sum\limits_{i=1}^N w_{mi} e^{-f(x) \alpha_{m+1} h^\star_{m+1}(x)}
\end{aligned}
$$

求损失函数$l(\alpha_{m+1}) = \sum\limits_{i=1}^N w_{mi} e^{-f(x) \alpha_{m+1} h^\star_{m+1}(x)}$对于$\alpha_{m+1}$的导数，可以得到：

$$
\begin{aligned}
    \frac{\partial l(\alpha_{m+1})}{\partial \alpha_{m+1}} &= \sum\limits_{f(x_i) \ne h_{m+1}(x_i)} w_{mi}e^{\alpha_{m+1}} - \sum\limits_{f(x_i) = h_{m+1}(x_i)} w_{mi}e^{-\alpha_{m+1}}
\end{aligned}
$$

令导数$\frac{\partial l(\alpha_{m+1})}{\partial \alpha_{m+1}} = 0$可得：
$$
\begin{aligned}
    \sum\limits_{f(x_i) \ne h_{m+1}(x_i)} w_{mi}e^{\alpha_{m+1}} &= \sum\limits_{f(x_i) = h_{m+1}(x_i)} w_{mi}e^{-\alpha_{m+1}}\\
    e^{2\alpha_{m+1}} &= \frac{\sum\limits_{f(x_i) = h_{m+1}(x_i)} w_{mi}}{\sum\limits_{f(x_i) \ne h_{m+1}(x_i)} w_{mi}}
\end{aligned}
$$

容易发现，因为$w_{mi}$是经过了归一化的，因此$\sum\limits_{i=1}^N w_{mi} = 1$，那么可以得到$\sum\limits_{f(x_i) = h_{m+1}(x_i)} w_{mi} = 1 - \sum\limits_{f(x_i) \ne h_{m+1}(x_i)} w_{mi}$，这里令$\sum\limits_{f(x_i) \ne h_{m+1}(x_i)} w_{mi} = e_{m+1}$表示$h_{m+1}$的损失，那么$\sum\limits_{f(x_i) = h_{m+1}(x_i)} w_{mi}$可以表示为$1 - e_{m+1}$，代入上面的导数为0的推导，可以得到：

$$
\begin{aligned}
    e^{2\alpha_{m+1}} &= \frac{1 - e_{m+1}}{e_{m+1}}\\
    \Rightarrow \alpha_{m+1} &= \frac{1}{2}\ln{\frac{1 - e_{m+1}}{e_{m+1}}}
\end{aligned}
$$

因此AdaBoost的流程就很明确了：

1. 在数据分布$D_{m-1}$上训练基学习器$h_m(x)$
2. 计算$e_m = \sum\limits_{f(x_i) \ne h_m(x_i)} w_{mi}$
3. 确定当前基学习器的权重$\alpha_m = \frac{1}{2}\ln{\frac{1 - e_m}{e_m}}$
4. 得到新的学习器$H_m(x) = \sum\limits_{i=1}^m \alpha_i h_i(x)$
5. 如果学习器个数达到指定个数，则退出，否则使用新的学习器调整权重参数$w_{mi} = \frac{e^{-f(x_i)(H_m(x_i)}}{\sum\limits_{i=1}^N e^{-f(x_i)(H_m(x_i)}}$得到分布$D_m$，跳转到第一步，进行第$m+1$个基学习器的训练。

## 提升树（Boosting Decision Tree, BDT）
残差树是针对回归问题的一种boosting方法。其基学习器是基于CART算法的回归树（关于决策树的相关内容可以见我的另外一篇文章{% post_link 决策树总结 决策树总结 %}），模型依旧为加法模型、损失函数为平方函数、学习算法为前向分步算法。

第$m$个树模型基学习器可以表示为$T(x; \theta_m)$，得到前$m$个基学习器之后，残差树的预测函数可以看做：$H_m(x) = \sum\limits_{i = 1}^mT(x; \theta_i)$，那么下一个树模型$T(x; \theta_{m+1})$的目标损失函数可以写作：
$$
\begin{aligned}
L(y, T(x; \theta_{m+1})) &= (y - H_m(x) - T(x; \theta_{m+1}))^2\\
&= (r - T(x; \theta_{m+1}))^2
\end{aligned}
$$
这里的$r$表示上一次的残差，这也是残差树名字的由来，例如现在需要拟合的值为20，第一次残差树拟合的值为18，那么第二次拟合的目标值为上一次的残差：20-18=2。

## 梯度提升决策树（Gradient Boosting Decision Tree, GBDT）
在提升树的训练过程中，每次增加的决策树是以残差作为目标，并使用CART方法构造决策树，而在GBDT中，添加新的决策树是以损失函数在当前模型预测值下的负梯度作为目标，同样使用CART方法构造决策树。如果损失函数使用的是均方误差，那么新的决策树的优化目标其实和提升树类似。

其实严格来将，GBDT并不是去拟合梯度，而是在进行决策树构建时，尽量将梯度相近的样本划分到同一个叶节点。

例如对于损失函数$\sum\limits_{i=1}^N l(y_i, \hat{y}_i)$，其中$N$为样本个数，$y_i$表示第$i$个样本的目标值，$\hat{y}_i$表示对第$i$个样本的预测值，现在GBDT已经构造出了第$t$颗决策树，目前的预测函数表示为$F_t(x) = \sum\limits_{i=1}^t f_i(x)$，这里每个$f_i(x)$表示一个决策树，现在的损失函数表示为$\sum\limits_{i=1}^N l(y_i, F_t(x_i))$，现在要构造第$t+1$颗树，那么其目标很简单，就是进一步使得损失函数$\sum\limits_{i=1}^N l(y_i, F_t(x_i) + f_{t+1}(x_i))$最小，这里将损失函数进行一次泰勒展开，可以将损失函数近似为：

$$
\begin{aligned}
    \sum\limits_{i=1}^N [l(y_i, F_t(x_i)) + \frac{\partial l(y_i, F_t(x_i))}{\partial F_t(x_i)}f_{t+1}(x_i)]
\end{aligned}
$$

其中$l(y_i, F_t(x_i))$的值和$f_{t+1}$无关，因此$f_{t+1}$只关心下面的表达式是否能够进一步将损失函数减小：

$$
\sum\limits_{i=1}^N \frac{\partial l(y_i, F_t(x_i))}{\partial F_t(x_i)}f_{t+1}(x_i)
$$

但是上面这个函数没有最小值，而且损失函数的一次展开是有误差的，$f_{t+1}$的预测值越大，误差越大，不能按照最小化问题来处理，自然的一个想法是使得$f_{t+1}(x_i) = -\frac{\partial l(y_i, F_t(x_i))}{\partial F_t(x_i)}$，因此GBDT这里将其简化为拟合负梯度。

### XGBoost
XGBoost是GBDT的一种改进版本。

在已知第$t$颗决策树时，下一个决策树的损失函数为：$\sum\limits_{i=1}^N l(y_i, F_t(x_i) + f_{t+1}(x_i)) + \Omega(f_{t+1})$。

这里损失函数中新增了正则化项$\Omega(f_{t+1})$，可以表示为：$\Omega(f_{t+1}) = \gamma T + \frac{1}{2}\lambda\sum\limits_{j=1}^Tw^2_j$，其中$T$表示第$t+1$颗树中的叶节点个数，而$w_j$表示第$j$个叶节点的输出值。

在XGBoost中，对损失函数进行二次泰勒展开近似：

$$
\begin{aligned}
    &\sum\limits_{i=1}^N l(y_i, F_t(x_i) + f_{t+1}(x_i)) + \Omega(f_{t+1})\\
    &= \sum\limits_{i=1}^N l(y_i, F_t(x_i) + f_{t+1}(x_i)) + \gamma T + \frac{1}{2}\lambda\sum\limits_{j=1}^Tw^2_j\\
    &\simeq \sum\limits_{i=1}^N [l(y_i, F_t(x_i)) + \frac{\partial l(y_i, F_t(x_i))}{\partial F_t(x_i)}f_{t+1}(x_i) + \frac{1}{2}\frac{\partial^2 l(y_i, F_t(x_i))}{\partial^2 F_t(x_i)}f^2_{t+1}(x_i)] + \gamma T + \frac{1}{2}\lambda\sum\limits_{j=1}^Tw^2_j\\
    &=constant + \sum\limits_{i=1}^N [\frac{\partial l(y_i, F_t(x_i))}{\partial F_t(x_i)}f_{t+1}(x_i) + \frac{1}{2}\frac{\partial^2 l(y_i, F_t(x_i))}{\partial^2 F_t(x_i)}f^2_{t+1}(x_i)] + \gamma T + \frac{1}{2}\lambda\sum\limits_{j=1}^Tw^2_j\\
    &=constant + \sum\limits_{j=1}^T [\sum\limits_{i\in I_j}\frac{\partial l(y_i, F_t(x_i))}{\partial F_t(x_i)}w_j + \sum\limits_{i\in I_j}\frac{1}{2}\frac{\partial^2 l(y_i, F_t(x_i))}{\partial^2 F_t(x_i)}w_j^2] + \gamma T + \frac{1}{2}\lambda\sum\limits_{j=1}^Tw^2_j\\
    &=constant + \sum\limits_{j=1}^T[\sum\limits_{i\in I_j} g_i w_j + \frac{1}{2}(\sum\limits_{i\in I_j}h_i + \lambda)w^2_j] + \gamma T\\
    &=constant + \sum\limits_{j=1}^T[G_j w_j + \frac{1}{2}(H_j + \lambda)w^2_j] + \gamma T
\end{aligned}
$$
这里的化简过程中，$I_j$表示属于第$j$个叶节点的样本id的集合，$g_i$表示第$i$个样本在$H_t(x_i)$这个预测结果下的一阶导数：$\frac{\partial l(y_i, F_t(x_i))}{\partial F_t(x_i)}$，$h_i$表示第$i$个样本在$F_t(x_i)$这个预测结果下的二阶导数：$\frac{\partial^2 l(y_i, F_t(x_i))}{\partial^2 F_t(x_i)}$，$G_j$表示属于第$j$个叶节点的样本的导数之和，$H_j$表示属于第$j$个叶节点的样本的二阶导数之和。

得到上面的化简结果之后，首先对$w_j$做个处理，令上式对$w_j$的导数为0，可以得到$w^\star_j = -\frac{G_j}{H_j + \lambda}$就是第$j$个叶节点的预测值的最优解，因此上式可以进一步写成：

$$
constant - \sum\limits_{j=1}^T \frac{1}{2}\frac{G^2_j}{H_j + \lambda} + \gamma T
$$

XGBoost在构造新的决策树时，和CART类似，将当前节点分裂成两个子节点，但是选择特征以及特征的最优划分，不是使用Gini_index，为了最小化上面的目标，XGBoost在划分新的叶节点时，判断是否将叶节点$j$划分为新的左节点和右节点时，主要考察下式：
$$
\begin{aligned}
&constant - \frac{1}{2}\frac{G^2_j}{H_j + \lambda} + \gamma T - (constant - \frac{1}{2}\frac{G^2_L}{H_L + \lambda} - \frac{1}{2}\frac{G^2_R}{H_R + \lambda} + \gamma(T+1))\\
&= \frac{1}{2}( \frac{G^2_L}{H_L + \lambda} + \frac{G^2_R}{H_R + \lambda} - \frac{G^2_j}{H_j + \lambda}) - \gamma    
\end{aligned}
$$

这里$G_L， H_L$分别表示划分出的左叶节点的导数和二阶导数和，$G_R， H_R$分别表示划分出的右叶节点的导数和二阶导数和。上式的结果表示将叶节点$j$划分为新的左节点和右节点时的增益情况。如果上式为正，则表示新划分节点有助于降低损失函数值，所以这里创建新节点时，遍历所有特征以及特征的划分位置，找到使得上式为正且最大的划分来创建新节点，如果没有能够使得上式为正的划分，那么停止继续添加叶节点。

按照上面的方法，逐个构造新的决策树以降低损失函数值，就是XGBoost的主要工作原理。
