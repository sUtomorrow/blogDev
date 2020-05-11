---
title: CNN权重初始化
date: 2020-05-11 16:00:57
tags: [深度学习]
mathjax: true
---

# 权重初始化

## 权重初始化的意义
CNN在训练开始之前，首先需要进行权重的初始化，如果初始化搞得好，例如直接初始化到最优点，那么训练步骤就可以省略了，当然这个概率基本为零，但是初始化如果做得好，跳开一些局部最优点也是可能的，这就是为什么神经网络如果不固定参数的初始化，每次训练得到的结果可能差别很大的原因。

初始化这里有个要注意的是：所有权重不能同时初始化为0。因为CNN中的反向传播如下所示，具体推导和含义可以查看我的另一篇文章{% post_link CNN的反向传播 CNN的反向传播 %}。如果全都初始化为0，那么所有的$\delta_i$全是0，从下面的式子可以看出，$\frac{\partial E}{\partial w_i}$和$\frac{\partial E}{\partial b_i}$恒为零，也就导致会导致学习到的权重完全一样，那么模型的众多参数就失去了意义。
$$
\begin{aligned}
    \delta_l = \frac{\partial E}{\partial z_l} &= diag(\sigma_l'(z_l))E'\\
    \delta_{i-1} &= \begin{cases}
    diag(\sigma_i'(z_{i-1}))w_i^T\delta_i &如果第i层是全连接层\\
    padpad(\delta_i) \star rot_{180}trans(w_i) \odot\sigma'(z_{i-1}) &如果第i层是卷积层\\
    M\odot upsample(\delta_i) &如果第i层是池化层
    \end{cases}\\
    \frac{\partial E}{\partial w_i} &= \begin{cases}
    \delta_il_{i-1}^T &如果第i层是全连接层\\
    padpad(\delta_i) \star rot_{180}trans(l_{i-1}) &如果第i层是卷积层\\
    \end{cases}\\
    \frac{\partial E}{\partial b_i} &= \delta_i,\ i=1,2,...,l
\end{aligned}
$$

另外，初始化的选择还和激活函数有关，如果使用ReLU激活函数，那么如果初始化权重让ReLU函数的输入全是小于0的，那得到的输出也是全0，并且不能进行有效的更新（ReLU死节点），如果使用Sigmoid激活函数，那么初始化权重如果使得Sigmoid函数的输入值偏离原点太远，也会导致梯度非常小（Sigmoid函数饱和）。

一般在初始化时，选择随机初始化方式，目前主流的初始化方式有：随机分布的初始化、Xavier、MSRA等。

## 随机分布的初始化
一般选择Gaussian分布或者均匀分布，这个没啥技术含量，主要是让均值和方差在合适的位置就好（参考前面对于不同激活函数的初始化要求的分析）。

## Xavier初始化
Xavier初始化来源于论文《Understanding the difficulty of training deep feedforward neural networks》，其中以tanh为激活函数，对权重初始化的方差提出了一些要求。

首先从正向传播过程来看：

如果第$i$层输入的方差是$\sigma^{i-1}_x$，权重初始化的方差$\sigma^i_w$，如果是卷积层，那么对于输出特征图的一个像素$z^{i+1}_j = \sum\limits_{k=1}^n x^{i-1}_k \times w^i_k$，其中$n = channel_{in} \times k_w \times k_h$，即输入通道数乘以卷积核面积。因为$x_i$，$w_i$相互独立（方差的和等于和的方差，方差的乘积等于乘积的方差），因此方差$\sigma^i_z = n^i \times \sigma^i_w \times \sigma^{i-1}_x$，因为tanh激活函数在原点附近区域可以近似为$f(x) = x$这样一个函数，所以论文中假设$x^i = z^i$，所以根据上面的分析，$\sigma^i_x = \sigma^i_z = n^i \times \sigma^i_w \times \sigma^{i-1}_x$，进一步可以得到：

$$
\begin{aligned}
    \sigma^i_x = \sigma^0_x\prod\limits_{k=1}^i(n^k \times \sigma^k_w)
\end{aligned}
$$

其中$\sigma^0_x$为模型的输入。从这个结果看，如果每一层的$\sigma^k_w$过大，则会引起深层的方差爆炸，如果过小，又会引起深层的方差消失。如果想要保证每一层的输入和输出的方差基本一致，需要$\sigma^k_w = \frac{1}{n^k}$。

其次从反向传播过程来看:

根据最前面列出的CNN反向传播过程，如果假设激活函数大致等价于$f(x) = x$，那么激活函数的导数$\sigma'(z_{i-1}) = 1$，因此同样有类似的结论$Var(\frac{\partial E}{\partial w_i}) = Var(\frac{\partial E}{\partial w_j}) \prod\limits_{k=i}^{j-1}(m^{k+1} \sigma^k_w)$，其中$m^{k+1}$表示第$k+1$层的每个输入值连接的权重个数，如果第$k+1$层是卷积层，则$m = channel_{out} \times k_w \times k_h$。

同样，如果要保证反向传播时梯度方差基本不变，则需要满足$\sigma^k_w = \frac{1}{m^{k+1}}$

因此根据以上前向传播和反向传播过程的分析，论文均衡了两个分析的结果，提出了Xavier初始化：
$$
\sigma^k_w = \frac{2}{m^{k+1} + n^k}
$$

如果使用均匀分布初始化，因为要满足权重分布在0附近的假设（否则上面的假设激活函数等价于$f(x) = x$不成立），我们选择$[-a, a]$范围的均匀分布，其方差$\frac{a^2}{3}$需要满足：

$$
\frac{a^2}{3} = \sigma^k_w = \frac{2}{m^{k+1} + n^k}
$$

可以得出$a = \sqrt{\frac{6}{m^{k+1} + n^k}}$，因此Xavier初始化建议使用$[-\sqrt{\frac{6}{m^{k+1} + n^k}}, \sqrt{\frac{6}{m^{k+1} + n^k}}]$范围的均匀分布，其中$m^{k+1}$表示第$k+1$层的每个输入值的连接个数，$n^k$表示第$k$层每个输出值的连接个数。

## MSRA初始化（一些深度学习框架中称为he_normal）
在论文 《Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification》中提出了PReLU激活函数和MSRA初始化方法。

MSRA初始化和Xavier初始化的动机类似，不过MSRA初始化是对于ReLU（或者PReLu）激活函数在原点附近进行分析，且使用Gaussian分布。

因为是ReLU激活函数，和上面Xavier初始化的分析类似，从正向传播的角度可以得到$\sigma^k_w = \frac{2}{n^k}$，从反向传播的角度可以得到$\sigma^k_w = \frac{2}{m^{k+1}}$。（这里的分析使用了x为0均值的假设）

如果是PReLU，则变成$\sigma^k_w = \frac{2}{(a^2 + 1)n^k}$和$\sigma^k_w = \frac{2}{(a^2 + 1)m^{k+1}}$，$a$为PReLU负区间的斜率。

因此MSRA初始化方法建议使用均值为0，方差为$\sigma^k_w = \frac{2}{(a^2 + 1)n^k}$或者$\sigma^k_w = \frac{2}{(a^2 + 1)m^{k+1}}$的Gaussian分布初始化。这里没有像Xavier初始化那样使用两个推理结果的折中，论文中说两个初始化方式差不多，都能够使模型收敛。
