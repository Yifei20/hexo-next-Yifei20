---
title: Test
abbrlink: bed2d29b
date: 2023-08-31 02:52:21
# description: ML Notes 1
tags: 
- 测试标签1
- 测试标签2
categories: 测试类1
---

![](https://cdn.jsdelivr.net/gh/Yifei20/blog-resource-bed/img/test-pic.JPG)

以根据前一天观看人数，预测第二天观看人数为例（Regression Example）

1. 根据 domain knowledge 提出一种合适的**函数/模型 (Model)**

   如 $Model:y=b+wx_i$，此处

   - $x_i$ 是 feature 即前一天的观看人数，为输入量

   - weight $w$ 和 bias $b$ 都是未知参数，需要*从数据中学习而来*

2. 定义合适的**损失函数 (Loss function)**

   如 $Loss=L(b,w)$，是以模型未知参数为输入的函数，用来评估这组参数的效果好坏。即，当取$L(0.5k,1)$时，此时 $Model: y=0.5k+1x_i$，其对应的损失函数为

   ​		Loss：$L=\frac{1}{n}\displaystyle\sum_{i=1}^{n}e_i$

   其中$e_i$为自定义的误差，用来衡量真实与估计值的差距，可以有不同计算方法，如

   - Mean Absolute Error (MAE): $e_i=|y-\hat{y}|$
   - Mean Square Error (MSE): $e_i=(y-\hat{y})^2$

   其中 $y=0.5k+1x_i$，而 $\hat{y}$ 为与给定输入 $x_i$ 对应的*真实值*

3. 优化（Optimization）

   使用**梯度下降（Gradient Descent）**方法根据**损失函数**优化模型参数

   对于只有一个未知参数的损失函数，如 $y=wx_i$，我们的目的是求 $w^*=arg\,\text{min}_w\,L$，即求能使L最小的 $w$，称为 $w^*$

   - 随机选择初始值 $w^0$

   - 计算在初始值处的微分 $\frac{\partial L}{\partial w}|_{w=w^0}$

   - 更新 $w$ 的值 $w^1=w^0-\eta\frac{\partial L}{\partial w}|_{w=w^0}$

```python
print(s)
```