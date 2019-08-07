---
title: “论文阅读《CornerNet-Detecting-Objects-as-Paired-Keypoints》”
date: 2019-08-07 20:17:13
tags: 论文阅读
mathjax: true
---

## 提出的问题

单阶段目标检测大多基于anchor实现，但是用anchor存在两个问题：

- 为了保证所有ground truth都有匹配的anchor，不得不使用非常多的anchor，例如retinanet anchor个数100K, DSSD anchor个数40K，但是真正与ground truth iou较大的anchor非常少，导致了很严重的正负样本不平衡问题。
- 设置anchor时需要非常多的超参数，anchor的大小、长宽比、个数，使得找到合适的anchor设置非常困难

## 解决思想
丢弃anchor，模型在回归目标位置时，不是去预测基于anchor的偏移量，而是使用两个feature map同时预测两个角点（左上、右下）即可，在feature map上的点，只有为目标角点的时候才进行激活预测其目标类别，只是这里又有三个问题：

- 两个feature map预测出来的角点，如何进行匹配，从而合并成一个目标？
- 目标在两个角点之间，因此feature map上的角点位置缺少关于目标的信息，如何准确的进行分类？
- feature map在经过下采样之后，feature map上的点的位置要映射回原图，这个时候存在位置精度的损失，可能造成预测位置的偏差。

## 具体做法
针对第一个角点匹配的问题，这里引用了另外一篇论文的方法（《ssociative embedding: End-to-end learning forjoint detection and grouping》，具体我没看），在模型中，两个角点坐标的feature map都会额外预测一个一维的嵌入向量（embedding vector），希望当两个角点匹配时，两个角点位置预测的嵌入向量的距离尽可能接近，否则尽可能远离（**这里嵌入向量的想法没看得太懂，可能还要看看原论文才能理解，除此之外计算量貌似有点大**），因此定义了两个loss如下所示，其中$e_{t_k}$代表第k个目标左上角点的嵌入向量，$e_{b_k}$代表第k个目标右下角点的嵌入向量，$e_k = \frac{(e_{t_k} + e_{b_k})}{2}$，N代表目标的个数，$\Delta$在论文的所有实验中均为1。
$$
\begin{aligned}
    L_{pull} &= \frac{1}{N}\sum_{k=1}^N[(e_{t_k} - e_k) ^ 2 + (e_{b_k} - e_k) ^ 2]\\
    L_{push} &= \frac{1}{N(N - 1)}\sum_{k=1}^N\sum_{\begin{aligned}
        j = 1 \\
        j \neq k
    \end{aligned}}max(0, \Delta - |e_k - e_j|)
\end{aligned}
$$

针对第二个角点处没有目标信息的问题，论文中提出了Corner Pooling，如下图所示，在左上角角点预测过程中，进行pooling时，每个点水平向右和垂直向下搜索，找到最大的值，之后加起来作为这个点pooling之后的值，这样给出了当前点是否是角点的判断信息，并且引入了一个非常强的先验：一个点是否是左上角点，需要看其右边的信息和下面的信息来判断，如果是在预测右下角点的时候，Corner Pooling变为每个点水平向左和垂直向上搜索。

![CornerPooling示意图](CornerPooling.png)

这里为了加快Corner Pooling的计算，如果是左上角点的Corner Pooling，可以从右下点开始计算，$t_{ij}$和$l_{ij}$分别代表水平向右搜索和垂直向下搜索的结果，计算方式如下，最终pooling结果$p_{ij} = t_{ij} + l_{ij}$。
$$
\begin{aligned}
    t_{ij} &= \begin{cases}
        max(f_{t_{ij}}, t_{(i+1)j}) & if \quad i < H\\
        f_{t_{Hj}} & otherwise
    \end{cases}\\
    l_{ij} &= \begin{cases}
        max(f_{t_{ij}}, t_{i(j+1)}) & if \quad j < W\\
        f_{t_{iW}} & otherwise
    \end{cases}
\end{aligned}
$$


