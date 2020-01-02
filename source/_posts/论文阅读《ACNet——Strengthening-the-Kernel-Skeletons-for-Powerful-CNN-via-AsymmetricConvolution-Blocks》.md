---
title: >-
  论文阅读《ACNet——Strengthening the Kernel Skeletons for Powerful CNN via
  AsymmetricConvolution Blocks》
date: 2019-12-31 15:02:12
tags: 论文阅读
mathjax: true
---

## 主要工作
提出了一种非对称卷积块（Asymmetric Convolution Block），使用一维的非对称卷积来加强方形卷积。

## 具体实现
在训练过程中，将原始的$3 \times 3$卷积替换为一种非对称卷积块的结构，如下图所示。

![训练时的ACB模块](ACB_training.png)

在预测过程中，将ACB模块中的三个卷积核加起来，重新变成一个$3 \times 3$卷积，如下图所示。

![预测时的ACB模块](ACB_evaluating.png)

对于预测过程，这样做并不会引起任何其他开销。但是实验结果表明，这样做可以提升模型的效果。

看起来这个ACB模块和直接训练一个$3 \times 3$卷积没有什么区别，但是一个细节的地方是，在实现ACB时，三个卷积分支在卷积层之后都会有一个BN层，在预测过程，混合卷积层权重时，BN层权重也需要进行混合，如下图所示。
![混合操作](BN_fusion.png)

那么这样分析之后，结果就很明显了，这就相当于为卷积核不同位置分配了可训练的权重。

## 我的实验结果
我自己在最近做的乳腺肿块分割任务上，对我的baseline模型使用了ACB，发现dice score确实有1个百分点的提升。

我还尝试直接为卷积核构造一个空间权重参数，但是发现效果反而不如直接使用ACB，可能是因为ACB只为十字上的卷积核参数分配权重，算是一种约束条件，更加容易优化吧。