## Pytorch 教程

> - https://www.bilibili.com/video/BV1Wv411h7kN?p=6 p6-p7 
>
> - https://gitee.com/Harris-X/PyTorch-Tutorial 
>
> - https://gitee.com/Harris-X/upc-cv-cource 
>
> - https://pan.baidu.com/s/1E5U2iFZjkBQJslhIReYgZg?pwd=u8yr  提取码：u8yr

[TOC]



### 0. 流程

<img src="Pytorch 教程.assets/image-20231015105354895.png" alt="image-20231015105354895" style="zoom:33%;" />

<img src="Pytorch 教程.assets/image-20231015105454691.png" alt="image-20231015105454691" style="zoom:40%;" />

### 1. 数据集处理

<img src="Pytorch 教程.assets/image-20231015105658446.png" alt="image-20231015105658446" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015105733805.png" alt="image-20231015105733805" style="zoom: 40%;" />

注意：index相当于给 一个数 选择样本。以下是一个batch的合成过程

<img src="Pytorch 教程.assets/image-20231015105901481.png" alt="image-20231015105901481" style="zoom:40%;" />

### 2. Tensors

#### 2.1 shape 查看维度

<img src="Pytorch 教程.assets/image-20231015110229592.png" alt="image-20231015110229592" style="zoom:40%;" />

#### 2.2 创建Tensors

<img src="Pytorch 教程.assets/image-20231015110434545.png" alt="image-20231015110434545" style="zoom:40%;" />

#### 2.3 基本操作

<img src="Pytorch 教程.assets/image-20231015110505184.png" alt="image-20231015110505184" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015110518786.png" alt="image-20231015110518786" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015110601613.png" alt="image-20231015110601613" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015110626579.png" alt="image-20231015110626579" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015110639989.png" alt="image-20231015110639989" style="zoom:40%;" />

#### 2.4 数据类型

<img src="Pytorch 教程.assets/image-20231015111834532.png" alt="image-20231015111834532" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015111855741.png" alt="image-20231015111855741" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015111943506.png" alt="image-20231015111943506" style="zoom:40%;" />

#### 2.5 运行设备

<img src="Pytorch 教程.assets/image-20231015112023446.png" alt="image-20231015112023446" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015112110052.png" alt="image-20231015112110052" style="zoom:40%;" />

#### 2.6 梯度计算

<img src="Pytorch 教程.assets/image-20231015112234681.png" alt="image-20231015112234681" style="zoom:40%;" />



### 3. 模型定义

#### 3.1 全连接层

<img src="Pytorch 教程.assets/image-20231015112818565.png" alt="image-20231015112818565" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015112849472.png" alt="image-20231015112849472" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015112905028.png" alt="image-20231015112905028" style="zoom:40%;" />

#### 3.2 激活函数-非线性

<img src="Pytorch 教程.assets/image-20231015113014697.png" alt="image-20231015113014697" style="zoom:40%;" />

#### 3.3 定义模型

<img src="Pytorch 教程.assets/image-20231015113055160.png" alt="image-20231015113055160" style="zoom:40%;" />

#### 3.4 损失函数

<img src="Pytorch 教程.assets/image-20231015113347927.png" alt="image-20231015113347927" style="zoom:80%;" />

#### 3.5 定义优化器

<img src="Pytorch 教程.assets/image-20231015113533622.png" alt="image-20231015113533622" style="zoom:80%;" />

<img src="Pytorch 教程.assets/image-20231015113552595.png" alt="image-20231015113552595" style="zoom:80%;" />

### 4. 模型训练

#### 4.1 训练设置

<img src="Pytorch 教程.assets/image-20231015113643221.png" alt="image-20231015113643221" style="zoom: 40%;" />

#### 4.2 训练循环

<img src="Pytorch 教程.assets/image-20231015113723761.png" alt="image-20231015113723761" style="zoom: 40%;" />

### 5. 模型验证

<img src="Pytorch 教程.assets/image-20231015113812633.png" alt="image-20231015113812633" style="zoom: 40%;" />

### 6. 模型测试

<img src="Pytorch 教程.assets/image-20231015113847084.png" alt="image-20231015113847084" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015113913082.png" alt="image-20231015113913082" style="zoom: 40%;" />

### 7. 模型保存

<img src="Pytorch 教程.assets/image-20231015114004522.png" alt="image-20231015114004522" style="zoom: 40%;" />

### 8. Demo

<img src="Pytorch 教程.assets/image-20231015114432055.png" alt="image-20231015114432055" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015114256152.png" alt="image-20231015114256152" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015114518984.png" alt="image-20231015114518984" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015114544561.png" alt="image-20231015114544561" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015114620172.png" alt="image-20231015114620172" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015114633362.png" alt="image-20231015114633362" style="zoom:40%;" />

<img src="Pytorch 教程.assets/image-20231015114649974.png" alt="image-20231015114649974" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015114658554.png" alt="image-20231015114658554" style="zoom: 40%;" />

### 9. Pytorch 常见错误

#### 9.1 Tensor on Different Device to Model

<img src="Pytorch 教程.assets/image-20231015115459577.png" alt="image-20231015115459577" style="zoom: 40%;" />

#### 9.2 Mismatched Dimensions

<img src="Pytorch 教程.assets/image-20231015115641807.png" alt="image-20231015115641807" style="zoom: 40%;" />

#### 9.3 Cuda Out of Memory

<img src="Pytorch 教程.assets/image-20231015115708432.png" alt="image-20231015115708432" style="zoom: 40%;" />

<img src="Pytorch 教程.assets/image-20231015115816152.png" alt="image-20231015115816152" style="zoom: 40%;" />

#### 9.4 Mismatched Tensor Type

<img src="Pytorch 教程.assets/image-20231015115858038.png" alt="image-20231015115858038" style="zoom: 40%;" />

