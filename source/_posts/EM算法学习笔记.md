---
title: EM算法学习笔记
date: 2020-01-02 15:27:25
tags: 机器学习
mathjax: true
---

## 问题描述
先看一个简单问题，例如现在有$m$个独立同分布样本$X=(x^{1}, x^{2}, \dots,x^{m})$，我们需要估计数据$X$的分布，那么我们只需要找到一个概率分布$P(x^{(i)}|\theta)$使得数据$X$的对数似然最大就行了，即只需要估计出参数$\theta=\mathop{\arg\max}_{\theta}L(\theta)=\mathop{\arg\max}_{\theta}\sum_{i=1}^{m}\log P(x^{(i)}|\theta)$

接下来问题变得复杂一些，还有一个不可观测的隐含变量$Z$会影响到$X$的分布，这个时候为了估计$X$的分布，所进行的工作将变为找到$\theta, Z = \mathop{\arg\max}_{\theta, Z}L(\theta, Z) = \mathop{\arg\max}_{\theta, Z}\sum_{i=1}^{m}logP(x^{(i)}|\theta) = \mathop{\arg\max}_{\theta, Z}\sum_{i=1}^{m}\log \sum_{z^{(i)}}P(x^{(i)}, z^{(i)}|\theta)$
这里$P(x^{(i)}|\theta)$其实是$P(x^{(i)}, z^{(i)}|\theta)$的边缘分布。

对于这个问题，直接对$Z$和$\theta$求导过于复杂，可能无法计算，因此需要EM算法来进行优化，得到一个可以接受的局部最优解。

## EM算法

EM算法分为两个重复的步骤，Expectation和Maximization

### Expectation

首先，重写上式如下，这里多出了一个关于$Z$的概率分布$Q$，对于$z^{(i)}$的所有取值，可知$\sum_{z^{(i)}}Q(z^{(i)}) = 1$。
$$
\begin{aligned}
&\mathop{\arg\max}_{\theta, Z}\sum_{i=1}^{m}\log \sum_{z^{(i)}}P(x^{(i)}, z^{(i)}|\theta)\\
&= \mathop{\arg\max}_{\theta, Z} \sum_{i=1}^{m}\log \sum_{z^{(i)}} Q(z^{(i)}) \frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})}\\
\end{aligned}
$$

由于$\sum_{z^{(i)}}Q(z^{(i)}) = 1$，因此$\log \sum_{z^{(i)}} Q(z^{(i)}) \frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})} = \log E(\frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})})$，又由于$\log$为凹函数，因此$\log E(\frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})}) \ge E(\log \frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})})$，即：

$$
\sum_{i=1}^{m}\log \sum_{z^{(i)}} Q(z^{(i)}) \frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})} \ge \sum_{i=1}^{m} \sum_{z^{(i)}} Q(z^{(i)}) \log \frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})}
$$

此时，假设$\theta$已经固定，那么要使得下面的等式成立

$\log E(\frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})}) = E(\log \frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})})$

我们需要让$\frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})}$等于一个常数$c$，因此对于所有的$i$，都满足

$c \times Q(z^{(i)}) = P(x^{(i)}, z^{(i)}|\theta)$

两边同时求和，得到

$c \times \sum_{z} Q(z^{(i)}) = \sum_{z} P(x^{(i)}, z^{(i)}|\theta)$

即：

$c = \sum_{z} P(x^{(i)}, z^{(i)}|\theta)$

因此：

$Q(z^{(i)}) = \frac{P(x^{(i)}, z^{(i)}|\theta)}{\sum_{z} P(x^{(i)}, z^{(i)}|\theta)} = \frac{P(x^{(i)}, z^{(i)}|\theta)}{P(x^{(i)}|\theta)} = P(z^{(i)}|x^{(i)},\theta)$

这样一来，确定了$Q(z^{(i)})$的值，等式可以成立，对数似然的下确界就被找到了，这个下确界即$\sum_{i=1}^{m} \sum_{z^{(i)}} Q(z^{(i)}) \log \frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})}$，接下来在Maximization步骤，固定$Q(z^{(i)})$，通过$\theta$来最大化这个下确界，可以间接的达到优化对数似然的目的。

### Maximization
由于$Q(z^{(i)}) = P(z^{(i)}|x^{(i)},\theta^{old})$被固定为常量，不再和$\theta$相关，因此问题变为
$$
\begin{aligned}
&\mathop{\arg\max}_{\theta} \sum_{i=1}^{m} \sum_{z^{(i)}} Q(z^{(i)}) \log \frac{P(x^{(i)}, z^{(i)}|\theta)}{Q(z^{(i)})}\\
&=\mathop{\arg\max}_{\theta} \sum_{i=1}^{m} \sum_{z^{(i)}} Q(z^{(i)}) \log P(x^{(i)}, z^{(i)}|\theta)
\end{aligned}
$$

总的来说，只需要首先初始化$\theta$，然后重复循环Expectation和Maximization两个步骤，不断更新$Q(z^{(i)})$和$\theta$，EM算法被证明最终可以收敛到一个局部最优解。
