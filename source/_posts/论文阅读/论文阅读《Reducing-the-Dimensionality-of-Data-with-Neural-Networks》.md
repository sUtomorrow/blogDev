---
title: 论文阅读《Reducing-the-Dimensionality-of-Data-with-Neural-Networks》
date: 2019-08-01 17:09:54
tags: 论文阅读
mathjax: true
---

## 提出的问题
使用多层神经网络进行数据降维编码和解码的时候，使用梯度下降法的效果非常依赖权重的初始化，难以训练。

## 解决思想
定义一种两层的神经网络模型（**注意，只有一层权重，但是有两个偏置变量**），称为限制玻尔兹曼机（RBM）,多个RBM串联组成一个自动编/解码器，训练时，每个RBM单独训练（每个玻尔兹曼机的目标都是使编码解码之后的输出和原始输入尽可能相同，具体模型结构图如下所示。
![RBM与编码/解码模型](RBM_and_multilayer_model.png)

## 实现细节
定义RBM的能量函数$E(v,h) = - \sum_{i \in pixels}b_i v_i - \sum_{j \in features}b_j h_j - \sum_{i,j}v_iw_{ij}h_j$，其中$v$表示可见单元(visible unit)，$h$表示隐藏单元(hidden unit)，$i$表示可见单元编号，$j$表示隐藏单元编号，$w_{ij}$表示权重矩阵中的相应值，$b_i$表示可见层的偏置，$b_j$表示隐藏层的偏置，训练每个RBM就是最小化其能量函数。