---
title: refineDet模型结构要点
date: 2019-04-15 09:21:39
tags: 深度学习
categories: 工程实践
---
因为需要实现keras版本的refineDet模型，因此考虑从keras-retinanet代码上进行修改，这里记录refineDet模型结构中的一些关键实现方式。

- 为了获取更高层的信息，在backbone网络中，加深了一层，对于ResNet-101，添加了一个额外的residual block.
- 在anchor大小选择方面，根据《S3FD: Single Shot Scale-invariant Face Detector》、《Understanding the Effective Receptive Field inDeep Convolutional Neural Networks》这两篇论文，anchor大小的选择应该要比理论感受小很多，使用有效感受野来估计每一个检测层的anchor大小，大约使用检测层stride的4倍作为anchor大小，并在anchor基础大小之上使用三种比例0.5,1.0,2.0
- anchor匹配时，首先对于每个ground truth，分别匹配一个jaccard overlap最大的anchor，之后再将剩下的每一个anchor分配给jaccard overlap大于0.5的ground truth。
- 计算loss时，对anchor进行hard negative mining，保证negative anchor和positive anchor的比例是3：1。