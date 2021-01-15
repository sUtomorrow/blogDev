---
title: 论文阅读《Multi-Task-Learning-Using-Uncertainty-to-Weigh-Lossesfor-Scene-Geometry-and-Semantics》
date: 2019-07-29 15:39:55
tags: 论文阅读
mathjax: true
---

## 提出的问题
在多任务学习的过程中，不同的损失函数权重设置非常困难。

## 解决思想
使用同方差不确定性来对多任务损失函数进行组合。

## 具体方法
$f^{W}(x)$代表输入x经过参数为$W$的神经网络之后得到的输出，那么对于回归任务，可以定义其高斯似然概率：$p(y|f^{W}(x)) = \mathcal{N}(f^{W}(x), \sigma^2)$,对于分类任务，因为常用Softmax来进行输出的处理，所以定义其似然概率为$p(y|f^{W}(x)) = Softmax(\frac{1}{\sigma^2}f^{W}(x))$，这里的$\frac{1}{\sigma^2}$不会影响Softmax的结果。

如果一个模型有多个任务，可以用最大似然法来定义其损失函数，假设模型存在一个回归任务和一个分类任务，其对数似然定义如下：
$$
\begin{aligned}
	log(p(y_1,y_2=c|f^{W}(x))) &= log(p(y_1|f^{W}(x))) + log(p(y_2=c|f^{W}(x), \sigma_2^2))\\
	&= log(\mathcal{N}(f^{W}(x), \sigma_1^2)) + log(Softmax(y_2=c; f^{W}(x)))\\
	&= log(\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x - f^{W}(x))^2}{2\sigma_1^2}}) + log(\frac{e^{\frac{1}{\sigma ^ 2}f_c^W(x)}}{\sum_{\hat{c}}e^{\frac{1}{\sigma_2 ^ 2}f_{\hat{c}}^W(x)}})\\
	&= -\frac{(x - f^{W}(x))^2}{2\sigma_1^2} - log\sigma_1 - log\sqrt{2\pi} + log(\frac{e^{\frac{1}{\sigma ^ 2}f_c^W(x)}}{\sum_{\hat{c}}e^{\frac{1}{\sigma_2 ^ 2}f_{\hat{c}}^W(x)}})\\
\end{aligned}
$$
当$\sigma_2$接近1时，$\frac{1}{\sigma_2}\sum_{\hat{c}}e^{\frac{1}{\sigma_2 ^ 2}f_{\hat{c}}^W(x)} \approx (\sum_{\hat{c}}e^{f_{\hat{c}}^W(x)})^\frac{1}{\sigma_2 ^ 2}$，上式可以写为：
$$
\begin{aligned}
	&\approx -\frac{(x - f^{W}(x))^2}{2\sigma_1^2} - log\sigma_1 - log\sqrt{2\pi} + log(\frac{e^{\frac{1}{\sigma ^ 2}f_c^W(x)}}{\sigma_2(\sum_{\hat{c}}e^{f_{\hat{c}}^W(x)})^\frac{1}{\sigma_2 ^ 2}})\\
	&= -\frac{(x - f^{W}(x))^2}{2\sigma_1^2} - log\sigma_1 - log\sqrt{2\pi} + \frac{1}{\sigma_2 ^ 2}log(\frac{e^{f_c^W(x)}}{\sum_{\hat{c}}e^{f_{\hat{c}}^W(x)}}) - log\sigma_2\\
	&= \frac{1}{\sigma_1 ^ 2}L_1(W) + \frac{1}{\sigma_2 ^ 2}L_2(W) - log\sigma_1 - log\sigma_2 - log\sqrt{2\pi}
\end{aligned}
$$

因此，在模型中设置两个可训练权重$\sigma_1$和$\sigma_2$，使用$-\frac{1}{\sigma_1 ^ 2}L_1(W) - \frac{1}{\sigma_2 ^ 2}L_2(W) + log\sigma_1 + log\sigma_2$作为多任务损失函数，即可自动学习模型的损失的权重。

## 一些问题
- **上式的简化是说在$\sigma_2$接近1的时候有用，但是模型学出来的权重很可能不在1附近，或者可以将$\sigma_2$固定为1？**
- **从论文贴出的结果看，组合多任务学习的效果比单任务的效果要差？结果图如下**

![论文中的实验结果对比](paper_result.png)
