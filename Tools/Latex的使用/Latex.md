# Latex的使用

> - 入门视频：<https://www.bilibili.com/video/BV1Mc411S75c>
> - 视频：<https://www.bilibili.com/video/BV1no4y1U7At>
> - [latex中文教程-15集从入门到精通包含各种latex操作](https://www.bilibili.com/video/BV15x411j7k6)
> - [Latex公式快速文档](https://mp.weixin.qq.com/s?__biz=MzAwOTM2NjU3MQ==&mid=2652275142&idx=4&sn=396dc796bc318634bd51db491aa22467&chksm=80829c82b7f5159497c07b4dd23df862ae240512b374fe33885dc67beab5430a8829c436f1c8&scene=27)
> - [Markdown/LaTeX 数学公式和符号表](https://zhuanlan.zhihu.com/p/450465546)

### 数学公式

- [多行公式的处理](https://www.bilibili.com/video/BV15x411j7k6?p=12&vd_source=29520f96e7e37ed65f945d56966cc4db)

$$
\begin{gather*}\label{equa2}
	&\alpha+\beta = \theta \notag \\
	&a+b=c
\end{gather*}
$$

$$
\begin{equation}
	\begin{split}
		\alpha+\beta &= \theta \\
		a+b&=c \\
		a &= \text{THU}^\text{E-ACT} \times 5
	\end{split}
\end{equation}
$$

$$
\begin{equation}
	\begin{cases}
		0&,\text{if}\qquad\alpha+\beta = \theta \\
		x+1&,\text{if}\qquad a+b=c \\
		1&,\text{if}\qquad a = \text{THU}^\text{E-ACT} \times 5
	\end{cases}
\end{equation}
$$

对齐符号`&`

- [LaTeX 数学公式：如何正确使用乘号](https://www.imooc.com/article/338918)
  - `\cdot`和`\ast`命令都表示乘号，但它们在某些情况下的表现略有不同。`\cdot`命令通常用于表示数字之间的乘法，而`\ast`命令有时用于表示向量或矩阵的乘法。

### 图片的插入

<img src="assets/image-20240711225852262.png" alt="image-20240711225852262" style="zoom:67%;" />

```
htbp是LaTX中用于控制浮动体位置的一个选项集。浮动体（如图片或表格）通常不会被直接放置在代码所在的位置，而是由LaTeX根据排版需要放置在页面的其他位置。htbp用于指定浮动体的偏好位置。这些选项的含义如下：
√ h（here):尽量将浮动体放置在代码所在的位置。然而，如果页面的顶部或底部能够更好地容纳浮动体，LaTeX可能会选择这样做。
√ t(op):将浮动体放置在页面的顶部。
√ b(bottom):将浮动体放置在页面的底部。
√ p（page):将浮动体放置在一个单独的页面上。
√ 这些选项可以组合使用，例如t表示首选放置在页面顶部，但如果不行就放置在代码所在的位置。
默认情况下，如果你不提供任何选项，LaTeX会使用tbp作为默认值。
√ 例如，begin{figure}[htbp]表示在尽量放在当前位置，如果不行就放在页面顶部，底部，或者单独一页。
```

### 伪代码

> - 快速入门：<https://blog.csdn.net/wangh0802/article/details/80788550>
> - 语法：<https://blog.csdn.net/kt1776133839/article/details/134419479>
> - 更多的案例：<https://blog.csdn.net/hl156/article/details/132085995>
> - 官方指导文件：<https://zhuanlan.zhihu.com/p/654374642>
>

一个实例

引入包

```
\usepackage{algorithm}  
\usepackage{algorithmic}  
```

伪代码

```
\begin{algorithm}[!h]
	\caption{PARTITION$(A,p,r)$}%算法标题
	\begin{algorithmic}[1]%一行一个标行号
		\STATE $i=p$
		\FOR{$j=p$ to $r$}
		\IF{$A[j]<=0$}
		\STATE $swap(A[i],A[j])$
		\STATE $i=i+1$
		\ENDIF
		\ENDFOR
	\end{algorithmic}
\end{algorithm}
```

### 表格

> - 快速上手：<https://blog.csdn.net/weixin_49322652/article/details/133375810>
> - <https://download.csdn.net/blog/column/11925364/129816219>
> - 博客：<https://blog.csdn.net/m0_71819746/article/details/135738780>
> - 生成工具：<https://www.latex-tables.com/>
> - [Latex 适配表格常用指令](https://blog.csdn.net/zjc910997316/article/details/117534795)
> - <https://blog.csdn.net/weixin_48958956/article/details/136423048>

- 一个demo

```
\begin{table}[!htbp]
\caption{\textbf{Classical table}}%title  
\centering  
\begin{tabular}{ccccc}% four columns  
\hline %begin the first line  
T 1 & T 2 & T 3 & T 4 \\  
\hline %begin the second line  
D1 & D2 & D3 & D4 \\  
D5 & D6 & D7 & D8 \\  
\hline %begin the third line  
\end{tabular}  
\end{table}
```

- [Latex 表格&公式大小问题](https://blog.csdn.net/weixin_41724971/article/details/136221296)

在LaTeX中，\setlength{\tabcolsep}{7mm}是一个命令，用于设置表格列之间的间距。tabcolsep是一个长度参数，它定义了表格中每一列两侧的空白距离。

通过使用\setlength{\tabcolsep}{7mm}命令，可以将表格列之间的间距设置为7毫米（mm）。这意味着在该表格中，每个单元格左右两侧将有7mm的空白距离，从而使表格看起来更加宽敞和美观。

这个命令通常用在tabular环境中，用于定制表格的样式和格式。通过调整tabcolsep的值，可以实现不同的列间距效果，以满足具体的排版需求。
