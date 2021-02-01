---
title: hexo安装记录
date: 2020-05-19 15:21:29
tags: [杂项]
---

# hexo安装过程
参考资料 https://zhuanlan.zhihu.com/p/187435941
## nodejs
版本：12.19.0，安装过程略

## 安装hexo命令
运行命令:

	 npm install -g hexo-cli

## 查看hexo版本
运行命令:

	 hexo --version

得到以下输出

	 hexo: 5.3.0
	 hexo-cli: 4.2.0
	 os: Windows_NT 10.0.19041 win32 x64
	 node: 12.19.0
	 v8: 7.8.279.23-node.44
	 uv: 1.39.0
	 zlib: 1.2.11
	 brotli: 1.0.9
	 ares: 1.16.0
	 modules: 72
	 nghttp2: 1.41.0
	 napi: 7
	 llhttp: 2.1.2
	 http_parser: 2.9.3
	 openssl: 1.1.1g
	 cldr: 37.0
	 icu: 67.1
	 tz: 2019c
	 unicode: 13.0

## 初始化博客文件夹
运行命令:

	 hexo init blogDev

## 安装package.json中的依赖

运行命令：

	 cd blogDev
	 npm install

## 修改_config.yml文件

使用hexo-autonofollow插件来给外链添加nofollow属性

在站点根目录下执行下列命令：

	 npm install hexo-autonofollow --save

## 修改git init文件
去掉以下三项：

	 db.json,
	 Thumbs.db
	 node-modules/

## 支持latex
卸载hexo自带的makrdown渲染工具：

	 npm uninstall hexo-renderer-markded --save

安装hexo-renderer-pandoc：

	 npm install hexo-renderer-pandoc --save

下载next主题：

	 git clone https://github.com/iissnan/hexo-theme-next themes/next

在_config.yml中将主题修改为next

修改next主题下的_config.yml文件中mathjax部分。

	 mathjax:
		enable: true
		per_page: false
		cdn: //cdn.bootcss.com/mathjax/2.7.1/latest.js?config=TeX-AMS-MML_HTMLorMML

在执行hexo s时遇到报错：

	 pandoc exited with code null

因此在 https://github.com/jgm/pandoc/releases 重新下载pandoc安装，这个问题解决后，访问网页时又遇到输出代码的问题，于是又安装swig：

	 npm i hexo-renderer-swig

网页可以正常访问，但是不能加载图片，又安装asset-image:

	 npm install hexo-asset-image --save

还需要在package.json中手动修改asset-image版本如下，并重新执行npm install：

	 "hexo-asset-image": "^0.0.1"


## 侧边栏中文乱码
是hexo-asset-image导致的，删除掉就好

## 设置首页不显示全文
在next的配置文件中，将下面的enable改成true

	 auto_excerpt:
		enable: false
		length: 150

## 设置导航栏
除了修改next主题下的config文件menu字段之外，还需要执行以下命令：

	 hexo new page "about"
	 hexo new page "tags"
	 hexo new page "categories"

## 翻页按钮显示不正常
修改代码的位置: themes\next\layout\_partials\pagination.swig


## Hexo next 主题配置右侧栏的分类和标签打开的是空白
categories 文件夹里面的 index.md 文件打开，修改（即添加一行）为：

	 ---
	 title: categories
	 date: 2018-01-23 17:14:51
	 type: "categories"   #新添加的
	 ---
同理，tags

	 ---
	 title: tags
	 date: 2018-01-23 17:14:51
	 type: "tags"     #新添加的
	 ---
