---
title: CBIB老服务器python环境配置
date: 2019-06-01 16:37:41
tags:
---

# python 虚拟环境配置步骤

## 说明

新的系统已经有公共的anaconda3环境,安装目录如下，因此无需重新安装anaconda，要创建自己的python环境，直接使用anaconda虚拟环境即可,下面介绍anaconda虚拟环境配置步骤。

     公共anaconda安装路径:/home/user/anaconda3/

## 环境变量配置
执行以下命令

     conda --version

若无报错，则无需更改环境变量，否则更改家目录下的 **.profile** 文件，在末尾添加如下内容：

     export PATH="/home/user/anaconda3/bin:$PATH"

使用 source 命令使其生效：

     source ~/.profile

之后即可使用conda命令

## conda 虚拟环境创建
执行如下命令，其中 **env name** 和 **python version**根据需求自行制定，执行后会在自己的 **~/.conda/envs**目录下创建虚拟环境。

     conda create -n <env name> python=<python version>

## 启用conda虚拟环境
在终端执行如下命令, 其中**env name**同上，即可启用上一步创建的虚拟环境

     source activate <env name>

## pip 安装package

启用conda虚拟环境之后，即可使用pip安装package，将默认优先安装到虚拟环境，不会影响公共环境。

