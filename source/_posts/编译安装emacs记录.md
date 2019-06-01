---
layout: w
title: 编译安装emacs记录
date: 2019-05-08 11:02:55
tags: 环境搭建
---

最近需要实现一些tensorflow OP，所以先找一个ubuntu下的C++ IDE，决定用emacs，下面是安装过程。

## 文件下载
使用镜像站: http://mirrors.kernel.org/gnu/emacs/，下载26.2版本

## 安装流程
按照安装说明，首先解包之后运行

     ./configure --prefix=/usr/local

遭遇无情报错：

{% asset_img configure_error.png configure提示环境问题%}

三个package没有，因此尝试安装
     sudo apt-get install libxpm-dev
     sudo apt-get install libdif-dev
     sudo apt-get install gnutls-bin
     sudo apt-get install libgnutls-dev

安装完后，configure就没有问题了,configure执行完后，按照常规操作，开始make，但是这里又遇到无情报错：

{% asset_img make_error.png make提示环境问题%}

locate 找了一下libpcre，发现就在Anaconda目录下面就有，因此make之前设置一下动态链接库搜索路径

     ecport LD_LIBRARY_PATH=/home/lty/env/Anaconda/lib:$LD_LIBRARY_PATH

重新make就成功了，make完成之后先检查编译出来的可执行文件是否可用，使用命令：

     src/emacs -Q

成功打开emacs，常规操作，没有什么问题，之后开始安装依旧是常规的 make install,