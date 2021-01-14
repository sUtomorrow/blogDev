---
title: CNN的反向传播
date: 2020-05-07 13:47:45
tags: [深度学习]
mathjax: true
---
在卷积神经网络中，主要由三种层结构构成：卷积层、池化层、全连接层。

# 全连接层的反向传播
全连接层的反向传播比较简单，使用单纯的BP算法即可，这里先来复习一下全连接层的BP算法：

对于$l$层神经网络,输入$x \in R^n$，标签$y \in R^c$，第$i$层权重表示为$w_i \in R^{O_i \times I_i}, I_1 = n，O_l = c$，第$i$层偏移表示为$b_i \in R^{O_i}$，第$i$层激活函数表示为$\sigma_i$，这一般是个逐元素函数，第$i$层输入即第$i-1$层的输出，表示为$l_{i-1}$，其中$l_0 = x, z_i = w_i l_{i-1} + b_i, l_i = \sigma_i(z_i)$

loss函数记为$E(l_l, y)$，BP算法每次更新$w_i = w_i - \eta \frac{\partial E}{\partial w_i}$， $b_i = b_i - \eta \frac{\partial E}{\partial b_i}$，即让参数像梯度最小的方向前进。

首先定义$E$对$l_l$的偏导为$\frac{\partial E}{\partial l_l} = E'$，这个值由loss函数决定。
因此
$$
\begin{aligned}
dE &= E'^Tdl_l\\
&=E'^T(\sigma_l'(z_l) \odot (dz_l))\\
&=E'^Tdiag(\sigma_l'(z_l))da_l\\
\Rightarrow \frac{\partial E}{\partial z_l} &= diag(\sigma_l'(z_l))E'
\end{aligned}
$$

这里把$\frac{\partial E}{\partial z_l}$记作$\delta_l$

因为：
$$
\begin{aligned}
    da_i &= w_idl_{i-1}\\
    &= w_i(\sigma_i'(z_{i-1}) \odot (dz_{i-1}))\\
    &=w_idiag(\sigma_{i-1}'(z_{i-1}))dz_{i-1}\\
    \Rightarrow \frac{\partial z_i}{\partial z_{i-1}} &= diag(\sigma_{i-1}'(z_{i-1}))w_i^T\\
\end{aligned}
$$

所以定义:
$$
\begin{aligned}
    \delta_i &= \frac{\partial E}{\partial z_i},\ i=1,2,...,l-1\\
    \Rightarrow \delta_{i-1} &= \frac{\partial z_i}{\partial z_{i-1}}\frac{\partial E}{\partial z_i},\ i=2,...,l\\
    &= diag(\sigma_{i-1}'(z_{i-1}))w_i^T\delta_i\\
\end{aligned}
$$

现在再来考虑$E$对$w_{l-k}$的导数：
$$
\begin{aligned}
    dE &= \frac{\partial E}{\partial z_{l-k}}^Tdz_{l-k}\\
    &= \delta_{l-k}^T(dw_{l-k}l_{l-k-1} + db_{l-k})\\
    &= tr(\delta_{l-k}^Tdw_{l-k}l_{l-k-1} + \delta_{l-k}^Tdb_{l-k})\\
    &= tr(l_{l-k-1}\delta_{l-k}^Tdw_{l-k} + \delta_{l-k}^Tdb_{l-k})\\
    \Rightarrow \frac{\partial E}{\partial w_{l-k}} &= \delta_{l-k}l_{l-k-1}^T\\
    \Rightarrow \frac{\partial E}{\partial b_{l-k}} &= \delta_{l-k}
\end{aligned}
$$
这里的变换属于标量对矩阵求导$d f = tr((\frac{\partial f}{\partial X}) ^ T dX)$，且用到了迹的一个性质：$tr(A B) = tr(B A)$，其中$A$和$B^T$大小相同

全连接层的BP算法看起来很复杂，其实非常简单，只要使用以下几个等式即可求出任一层的权重和偏置的导数：
$$
\begin{aligned}
    \delta_l = \frac{\partial E}{\partial z_l} &= diag(\sigma_l'(z_l))E'\\
    \delta_{i} = \frac{\partial E}{\partial z_i} &= diag(\sigma_i'(z_i))w_{i+1}^T\delta_{i+1},\ i=1,2,...,l-1\\
    \frac{\partial E}{\partial w_i} &= \delta_il_{i-1}^T,\ i=1,2,...,l\\
    \frac{\partial E}{\partial b_i} &= \delta_i,\ i=1,2,...,l
\end{aligned}
$$

# 卷积层的反向传播

对于卷积层而言，和前面的定义类似，只不过其表达式变为：$l_0 = x, z_i =l_{i-1} \star w_i + b_i, l_i = \sigma_i(z_i)$，其中$\star$表示卷积操作，$w_i \in R^{H\times W\times C_{i}\times C_{i-1}}$，$x\in R^{H\times W\times C_0}$，$z_i,l_i \in R^{H\times W\times C_i}$，$b_i \in R^{C_i}$。

按照之前全连接层的反向传播套路，自然也希望首先定义
$$
\begin{aligned}
    \delta_l &= \frac{\partial E}{\partial z_l} = diag(\sigma_l'(z_l))E'\\
    \delta_{i} &= \frac{\partial E}{\partial z_i}
\end{aligned}
$$
那么第一个问题是如何根据$\delta_{i+1}$求出$\delta_{i}$。

因为：
$$
\begin{aligned}
    \delta_{i} &= \frac{\partial E}{\partial z_i}\\
    &= \frac{\partial z_{i+1}}{\partial z_i}\frac{\partial E}{\partial z_{i+1}}\\
    &= \frac{\partial z_{i+1}}{\partial z_i}\delta_{i+1}
\end{aligned}
$$

所以问题在于如何求$\frac{\partial z_{i+1}}{\partial z_i}$，如果$i+1$层是个卷积层，那么$z_{i+1} = \sigma_i(z_i) \star w_{i+1} + b_{i+1}$，这里想使用一般的方法求解$\frac{\partial z_{i+1}}{\partial z_i}$是很困难的，下面直接对$\delta_{i}$给出一般化的表达。


卷积操作的具体分析需要画图讨论，这里不再赘述，定义一个卷积操作，$O = I \star W$，其中卷积核$W$的大小为$H_k\times W_k\times C_o\times C_i$，输入特征图的大小为$H\times W\times C_i$，卷积操作的padding大小为$P_H,P_W$，stride大小为$S_H,S_W$，则输出特征图的大小为$\lceil\frac{H + 2P_H - H_k + 1}{S_H}\rceil\times \lceil\frac{W + 2P_W - H_k + 1}{S_W}\rceil\times C_o$，注意这里$0 \le P_H \le H_k - 1$，$0 \le P_W \le W_k - 1$，否则过多的padding没有意义。

设L为损失函数，令$\frac{\partial E}{\partial O}$记作$\delta_O$，其大小和$O$相同。
$$
\begin{aligned}
    \frac{\partial E}{\partial I} &= padpad(\delta_O) \star rot_{180}trans(W)
\end{aligned}
$$
其中$padpad(\delta_O)$是将$\delta_O$进行外部padding和内部padding得到的，其外部padding大小：$P'_H = H_k - P_H - 1$和$P'_W = W_k - P_W - 1$，内部padding大小为$S_H - 1$和$S_W - 1$。

$rot_{180}trans(W)$则首先需要将$W$在$H_k\times W_k$大小上，旋转180度，然后对其中每个像素（其实每个像素都是个$C_o\times C_i$大小的矩阵）求转置得到，最终的形状是$H_k\times W_k\times C_i\times C_o$，其中$rot_{180}trans(W)_{i,j,k,l} = W_{H_k-i+1,W_k-j+1,l,k},\ i=1,2,...,H_k,\ j=1,2,...,W_k,\ k=1,2,...,C_o,\ l=1,2,...,C_i$

则有:
$$\delta_{i} = padpad(\delta_{i+1}) \star rot_{180}trans(w_{i+1}) \odot\sigma_{i}'(z_i)$$

其中$\odot$表示逐元素乘法，由于这里不是向量和向量乘积，因此不能像之前一样表示成$diag$矩阵乘以向量的形式。

$\delta_i,\ i=1,2,...,l$能够求出来了，接下来的问题是如何根据$\delta_i$求出$\frac{\partial E}{\partial w_i}$。

因为$a_i =l_{i-1} \star w_i + b_i$，将$a_i$、$l_{i-1}$和$w_i$都旋转180度之后，可以看成$rot180(a_i) = rot180(w_i) \star rot180(l_{i-1}) + b_i$，这里将$l_{i-1}$看成卷积核，卷积核大小$\hat{W}_k = W, \hat{H}_k = H$，Padding变成了$\hat{P}_H = H - H_k + P_H$和$\hat{P}_W = W - W_k + P_W$，stride不变，则按照上面的分析结果，对$rot_{180}(\delta_i)$和$rot_{180}(l_{i-1})$做同样的变换：

$$
\begin{aligned}
    \frac{\partial E}{\partial w_i} &= rot_{180}(padpad(rot_{180}(\delta_i)) \star rot_{180}trans(rot_{180}(l_{i-1})))\\
    &= rot_{180}(padpad(rot_{180}(\delta_i)) \star trans(l_{i-1}))\\
    &= padpad(\delta_i) \star rot_{180}trans(l_{i-1})
\end{aligned}
$$


其中$padpad(rot180(\delta_i))$是将$rot180(\delta_i)$进行外部padding和内部padding得到的，其外部padding大小：$P'_H = \hat{H}_k - \hat{P}_H - 1 = H - H + H_k - P_H - 1 = H_k - P_H - 1$和$P'_W = \hat{W}_k - \hat{P}_W - 1 = W_k - P_W - 1$，内部padding大小为$S_H - 1$和$S_W - 1$，可以看出来这里的padpad操作和之前的padpad操作是一样的，两个地方完美等价。

因此如果知道了$\delta_i$，那么：
$$
\frac{\partial E}{\partial w_i} = padpad(\delta_i) \star rot_{180}trans(l_{i-1})
$$

因此如果第$i$层是个卷积层，那么这一层的反向传播核心公式如下：
$$
\begin{aligned}
    \delta_{i-1} &= padpad(\delta_i) \star rot_{180}trans(w_i) \odot\sigma_{i-1}'(z_{i-1})\\
    \frac{\partial E}{\partial w_i} &= padpad(\delta_i) \star rot_{180}trans(l_{i-1})\\
    \frac{\partial E}{\partial b_i} &= \delta_{i}
\end{aligned}
$$

# 池化层的反向传播

池化层没有参数，如果第$i$层是池化层，其反向传播主要需要计算$\delta_i$和$\delta_{i-1}$的关系即可。

池化层的$\delta_i$和$\delta_{i-1}$的关系取决于池化的类型，如果是最大池化，则需要构造一个非0即1的掩码矩阵，用于标记哪些位置被向前传播，如果是平均池化，则将权重1平均分配到池化核大小的窗口中，以此来构造掩码矩阵

首先上采样$\delta_i$，使其和$\delta_{i-1}$的大小相同，然后根据池化类型，构造掩码矩阵$M$，则$\delta_{i-1} = M\odot upsample(\delta_i)$

# CNN反向传播的总结

对于共$l$层的CNN，如果第$i$层是全连接层，则其权重表示为$w_i \in R^{O_i \times I_i},O_l = c$，第$i$层偏移表示为$b_i \in R^{O_i}$，第$i$层激活函数表示为$\sigma_i$，这一般是个逐元素函数，第$i$层输入即第$i-1$层的输出，表示为$l_{i-1}$，其中$z_i = w_i l_{i-1} + b_i, l_i = \sigma_i(z_i)$

如果第$i$层是卷积层，则其表达式变为：$z_i =l_{i-1} \star w_i + b_i, l_i = \sigma_i(z_i)$，其中$\star$表示卷积操作，$w_i \in R^{H\times W\times C_{i}\times C_{i-1}}$，$x\in R^{H\times W\times C_0}$，$z_i,l_i \in R^{H\times W\times C_i}$，$b_i \in R^{C_i}$。

loss函数记为$E(l_l, y)$。

这里默认最后一层即第$l$层为全连接层。
$$
\begin{aligned}
    \delta_l = \frac{\partial E}{\partial z_l} &= diag(\sigma_l'(z_l))E'\\
    \delta_{i-1} &= \begin{cases}
    diag(\sigma_{i-1}'(z_{i-1}))w_i^T\delta_i &如果第i层是全连接层\\
    padpad(\delta_i) \star rot_{180}trans(w_i) \odot\sigma_{i-1}'(z_{i-1}) &如果第i层是卷积层\\
    M\odot upsample(\delta_i) &如果第i层是池化层
    \end{cases}\\
    \frac{\partial E}{\partial w_i} &= \begin{cases}
    \delta_il_{i-1}^T &如果第i层是全连接层\\
    padpad(\delta_i) \star rot_{180}trans(l_{i-1}) &如果第i层是卷积层\\
    \end{cases}\\
    \frac{\partial E}{\partial b_i} &= \delta_i,\ i=1,2,...,l
\end{aligned}
$$

其中$padpad$表示内部padding（列和行中每两个元素之间）：$(S_H - 1, S_W - 1)$和外部padding：$(H_k - P_H - 1, W_k - P_W - 1)$，$rot_{180}$表示在空间维度(H,W)上旋转180度，$trans$表示在通道维度上转置，$M$表示池化操作的掩码。

