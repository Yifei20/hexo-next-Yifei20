---
title: Artificial Neuron Network (1)
abbrlink: bed2d29b
date: 2023-08-31 02:52:21
tags: 
- ANN
- DL
- ML
categories: 学习笔记
---

## 1. Introduction to ANN

#### 1.1 Relationship between the key concepts

我们知道ANN是人工智能领域的一环，但是它具体在这个领域有什么样的作用呢？要回答这个问题，我们就需要搞清楚人工智能方面几个概念之间的关系，即包括：

- Artificial Intelligence (AI)
- Machine Learning (ML)
- Neuron Network (NN)
- Deep Learning (DL)

![relationship demonstration](https://cdn.jsdelivr.net/gh/Yifei20/blog-resource-bed/img/relationship-ai-key-concepts.png)

首先，AI（人工智能）是一个很大很笼统的概念，我们可以将很多事情称为人工智能，比如说让机器进行1+1的算数运算，虽然这是一种程度很低的人工智能，但它确实也能够模拟人类的一部分能力。

其次，ML（机器学习）是人工智能的一个子领域，其指的是，通过统计学方法或是数学模式，使得计算机能够从数据中进行学习改进，从而达到某种模拟人类的能力。

而，NN（神经网络）则是一类机器学习算法模型，其结构设计由早期的神经学家在解剖和生理学的的研究成果启发而来，其一开始的目的在于模拟大脑的神经结构。但是随着发展，神经网络的结构和真正的神经元除了名字，差别已经非常大了，而演化来的神经网络则被广泛应用到了机器学习中。

最后，DL（深度学习）是一类多层（Multi-Layer）的神经网络，与神经网络一样，是机器学习的一个子领域。

#### 1.2 Abstract neuron (from Biology to Computer Sicence)

我们在高中都学过神经元的基本结构和功能，这里我们基于神经元的结构和功能，尝试通过构建一个抽象神经元来模拟真实的神经元。

![abstract neuron](https://cdn.jsdelivr.net/gh/Yifei20/blog-resource-bed/img/abstract-neuron.png)

如上图所示，我们构建一个抽象神经元$j$。

1. 输入（Inputs）：神经元能够彼此接受信息，抽象神经元$j$中有$n$个输入$x_1,...,x_n$还有输出$o_j$。
2. 权重（Weights）：此外，神经元不同路径的输入的重要性可能不同，对应$n$条不同路径的不同权重$w_{j1},w_{j2},...,w_{jn}$。
3. 偏差（Bias）：通常神经网络模型还会有偏差（即bias，对应$x_0,w_{0j}$，其中$x_1$总为$+1$，而$w_{0j}$可在学习中改变）。
4. 再者，神经元会处理接受的信息，并判断如何做出反应。这对应的是：
   1. **处理输入信息：**通过转化函数（transfer function），将零散的输入数值转化成一个值，以进行下一步处理。这个转换函数接受加权后的输入，输出转化后的网络输入值（net input）$net_j$。 比如，一个简单的转化函数可以是求和函数$\sum$，将加权输入值求和传递到下一步。
   1. **决定输出：**使用一个激活函数（activation function）根据处理过的信息决定神经元的输出。一个简单的激活函数可以包括一个阈值（threshold）$\theta_j$，函数简单得将输入的$net_j$和阈值$\theta_j$进行比较，如果输入能够超过阈值，那么神经元被激活（activated），向下一个神经元传递信息；否则不做出行动。


## 2. Constructing ANN from the Beginning

基于前面的理解，我们开始从头构建神经网络的概念。首先我们从最简单的模型开始学起。*需要注意的是，虽然下面的不同模型直接有时间上的递进和概念上相对的继承关系，但实际上他们的概念互不隶属，注意区分理解。*

#### 2.1 The McCulloch-Pitts Neuron (1943)

>  为了方便讨论，这里只讨论仅存在唯一神经元的情况，故不对属性作神经元标记

##### 2.1.1 Basic definition

这种神经元模型的神经元有下面几个特点或属性：

1. 离散时间（Discrete-time）：神经元的输入输出等值具有离散的时间属性
2. 二元（Binary）：输入$a^t_i$输出$x^t$是二元的，即只有1和0两种情况。
3. 有激活和抑制权重（Excitatory and Inhibitory Weights）：权重有激活（excitatory, +1）和抑制（inhibitory, -1）两种情况。
4. 有激活阈值（Excitation Threshold）：每个神经元都有一个激活阈值$\theta$

##### ![mp_neuron demo](https://cdn.jsdelivr.net/gh/Yifei20/blog-resource-bed/img/mc_neuron%20demo.png)

##### 2.1.2 Calculating the output

对一个MP神经元，其$t$时刻的输入所对应的输出为$x^{t+1}$，当且仅当其状态达到阈值时输出为1，即
$$
x^{t+1}=
\begin{cases}
1\text{, iff }S^t=\sum_{i=1}^n a^t_iw_i\ge\theta\\
0\text{, otherwise}
\end{cases}
$$
神经元的**状态**（State）：我们称这里某时刻的加权输入总和（instant total sum）$S^t=\sum_{i=1}^n a^t_iw_i$，为 Instant **State** of the Neuron。神经元还有其他状态，如Previous State, etc..

##### 2.1.3 Mathematical Definition

**Write state as $f(t)$: **如前面所示，MP神经元某时刻的状态与其他时刻的值无关，仅与其所属时刻的输入和权重值有关，即$S^t=\sum_{i=1}^n a^t_iw_i=f(t)$。

**Write output as $g(f(t))$: **神经元输出$x^{t+1}$对应$t$时刻的函数可以写成：$x^{t+1}=x(t)=g(S^t)=g(f(t))$

**Heaviside Function: **这里的$g()$称作阈值激活函数（threshold activation function），用来判断给定状态是否满足输出条件。

对于MP神经元，$g()$实际上是一种 Heaviside (unit step) function，即$H(S^t-\theta)$
$$
H(x)=\begin{cases}1\text{, }x\ge0;\\0\text{, }x<0.\end{cases}
$$
![unit step function](https://cdn.jsdelivr.net/gh/Yifei20/blog-resource-bed/img/unit%20step%20func.jpg)