- basic：https://pkm-er.github.io/Pkmer-Math/%E6%A6%82%E7%8E%87%E8%AE%BA/%E5%8F%82%E6%95%B0%E4%BC%B0%E8%AE%A1/%E4%BC%BC%E7%84%B6%E5%87%BD%E6%95%B0.html#_2-%E5%AF%B9%E6%95%B0%E4%BC%BC%E7%84%B6%E5%87%BD%E6%95%B0

- https://blog.csdn.net/m0_57407372/article/details/144659244

### Fisher Information (费雪信息)

https://zhuanlan.zhihu.com/p/589273267

#### 定义

Fisher Information 是一种衡量“随机观测样本携带的未知参数θ的信息量”的方法，其中θ为待估计的参数。

假定观测随机变量序列为 $X_1, X_2, \ldots, X_3$，且都服从概率分布 $f(X; \theta)$，则似然函数 $Q$ 可以表示成：

$$L(\mathbf{X}; \theta) = \prod_{i=1}^{n} f(X_i; \theta)$$

对数似然函数 $Q$ 对 $\theta$ 求导并令一阶导数为0，则可以得到 $\theta$ 的最大似然估计 $\hat{\theta}$。上述对数似然函数的一阶导数也称作 Score function，其定义为：

$$S(\mathbf{X}; \theta) = \sum_{i=1}^{n} \frac{\partial \log f(X_i; \theta)}{\partial \theta} = \sum_{i=1}^{n} S(X_i; \theta)$$

那么 Fisher Information 定义为 Score function 的二阶矩 $I(\theta) = E \left[ S(\mathbf{X}; \theta)^2 \right]$，下面对以下两点进行证明：

- $E[S(\mathbf{X}; \theta)] = 0$
- $I(\theta) = E \left[ S(\mathbf{X}; \theta)^2 \right] = \text{Var}[S(\mathbf{X}; \theta)]$

# 费舍尔信息矩阵 (Fisher Information Matrix)

费舍尔信息矩阵是统计学中一个非常重要的概念，尤其在参数估计、最大似然估计 (MLE) 和贝叶斯推断中具有广泛的应用。它反映了参数估计的不确定性程度，也可以用来衡量数据提供了多少关于参数的信息。

## 1. 费舍尔信息的基本概念

在统计学中，给定一个模型 $Q$，模型的参数往往是我们感兴趣的未知量。费舍尔信息矩阵量化了模型参数的可估计性，即参数的估计值相对于真实值的精确度。费舍尔信息越大，表示数据对于估计这些参数的“信息”越多，估计的精度越高；反之，费舍尔信息越小，参数的估计就越不精确。

定义：费舍尔信息矩阵是基于对数似然函数的二阶导数的期望值。对于参数向量 $\theta = (\theta_1, \theta_2, \cdots, \theta_k)$，费舍尔信息矩阵 $I(\theta)$ 是一个 $k \times k$ 的矩阵，其中每个元素是参数对数似然函数的二阶偏导数的期望。

具体而言，假设有一个观测数据集 $X = (X_1, X_2, \cdots, X_n)$，其联合概率密度函数（或概率质量函数）为 $p(X; \theta)$，其中 $\theta$ 是待估计的参数，费舍尔信息矩阵的定义为：

$$I(\theta) = -E\left[\frac{\partial^2}{\partial \theta^2} \log p(X; \theta)\right]$$

其中，$E$ 是对数据的期望，$\log p(X; \theta)$ 是对数似然函数，$\frac{\partial^2}{\partial \theta^2}$ 是对数似然函数关于参数 $\theta$ 的二阶导数。

## 2. 费舍尔信息矩阵的含义

- **信息量**：费舍尔信息度量了数据对于估计某个参数的“信息量”。如果费舍尔信息较大，意味着观测数据对于该参数的估计越精确。
- **不确定性**：费舍尔信息的倒数是参数估计的方差的下界，即 Cramér-Rao 下界（Cramér-Rao Bound）。根据 Cramér-Rao 不等式，参数的无偏估计量的方差不能小于费舍尔信息的倒数。因此，费舍尔信息矩阵提供了对参数估计方差的下限约束。

$$Var(\hat{\theta}) \geq (I(\theta))^{-1}$$

- 这里，$\hat{\theta}$ 是参数的估计值，$I(\theta)$ 是费舍尔信息矩阵。

## 3. 费舍尔信息矩阵的数学表达

对于一个模型，假设样本 $X$ 的联合概率密度函数为 $p(X; \theta)$，其中 $\theta$ 为参数，$logp(X; \theta)$ 是对数似然函数。费舍尔信息矩阵的元素可以通过以下公式计算：

$$I_{ij}(\theta) = -E\left[\frac{\partial^2}{\partial \theta_i \partial \theta_j} logp(X; \theta)\right]$$

其中，$i$ 和 $j$ 表示参数的不同维度，$\theta_i$ 和 $\theta_j$ 是参数的不同分量。具体来说，费舍尔信息矩阵中的每个元素 $I_{ij}$ 表示参数 $\theta_i$ 和 $\theta_j$ 对似然函数的二阶导数的期望值。

## 4. 费舍尔信息矩阵的性质

- **对称性**：费舍尔信息矩阵是对称矩阵，即 $I_{ij}(\theta) = I_{ji}(\theta)$。这是由于对数似然函数的二阶偏导数是对称的。
- **正定性**：费舍尔信息矩阵是正定的，即它的特征值全为正。这意味着它的逆矩阵（即Cramér-Rao下界）存在，并且可以用于描述参数估计的精确度。
- **无偏估计**：根据Cramér-Rao下界，若参数的估计量是无偏的，那么它的方差的下界由费舍尔信息矩阵的逆给出。

## 5. 计算例子

假设我们要估计一个正态分布的均值和方差，数据为 $X_1, X_2, \cdots, X_n$，假设数据来自正态分布 $N(\mu, \sigma^2)$，其中 $\mu$ 和 $\sigma^2$ 是需要估计的参数。

1. **对数似然函数**：正态分布的概率密度函数为：

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

对其取对数，得到对数似然函数：

$$logL(\mu, \sigma^2) = -\frac{n}{2}log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

2. **计算费舍尔信息矩阵**：计算对数似然函数关于 $\mu$ 和 $\sigma^2$ 的二阶偏导数并求期望。通过这些步骤，我们可以得到每个参数的费舍尔信息。

最终得到的费舍尔信息矩阵的形式是一个 2×2 的矩阵，包含对均值和方差的估计不确定性的描述。