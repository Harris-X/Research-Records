# WUDI-Merging方法详细解析（基于2503.08099v2.pdf）
WUDI-Merging（Whoever started the interference shoUld enD It）是一种**无数据模型融合方法**，核心是通过“任务向量指导干扰最小化”解决专家模型间的参数干扰问题。其设计完全依赖文档中的理论推导（输入一致性、任务向量线性子空间性质、干扰上界），以下从**符号体系、理论推导（含核心公式）、方法优化目标**三部分展开，严格遵循文档定义与逻辑。


## 一、核心符号体系定义
文档中所有公式与推导均基于统一的符号体系，需先明确关键符号的物理意义（下表覆盖90%以上核心符号）：

| 符号                | 类型          | 文档定义（出处：3.1节、A.1节）                                                                 | 通俗解释                                                                 |
|---------------------|---------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| $\theta$          | 模型参数      | 预训练模型的全部参数（所有层共享该基准）                                                       | 微调的“起点”参数，如ViT-B/32的预训练权重                                 |
| $\theta_i$        | 模型参数      | 任务$i$的专家模型参数（由$\theta$微调得到）                                               | 适配特定任务$i$的最终参数                                               |
| $\theta_{i,l}$    | 层参数        | 专家模型$i$的第$l$个线性层参数                                                             | 仅关注线性层（文档证明其任务向量保留模型核心能力）                       |
| $\theta_m$        | 模型参数      | 融合模型的全部参数                                                                             | 最终输出的多任务模型参数                                                 |
| $\theta_{m,l}$    | 层参数        | 融合模型的第$l$个线性层参数                                                                 | 逐层优化的核心对象                                                       |
| $\tau_i$          | 任务向量      | $\tau_i = \theta_i - \theta$（专家参数与预训练参数的差值）                                  | 量化专家模型为适配任务$i$的“参数更新量”                               |
| $\tau_{i,l}$      | 层任务向量    | $\tau_{i,l} = \theta_{i,l} - \theta_l$（线性层$l$的任务向量）                             | 线性层$l$适配任务$i$的参数更新量                                     |
| $\tau_m$          | 融合任务向量  | 融合模型的整体任务向量，$\theta_m = \theta + \tau_m$                                        | 融合模型的“总参数更新量”，由各线性层的$\tau_{m,l}$组装而成             |
| $\tau_{m,l}$      | 层融合向量    | 融合模型线性层$l$的任务向量                                                                 | 每层独立优化的目标变量                                                   |
| $\delta_{i,l}$    | 干扰向量      | $\delta_{i,l} = \tau_{m,l} - \tau_{i,l}$（融合向量与专家向量的差值）                        | 导致融合模型性能下降的“干扰来源”                                         |
| $x_{i,l}$         | 输入向量      | 任务$i$的数据经过前$l-1$层后，输入到线性层$l$的特征向量                                 | 线性层$l$的“输入信号”，是干扰作用的载体                               |
| $x_{n,l}^t$       | 输入向量      | 微调第$t$次迭代时，第$n$个样本在 linear layer $l$ 的输入                                 | 动态跟踪迭代过程中的输入变化                                             |
| $f_{\sim l}(x;\theta_{\sim l})$ | 映射函数 | 前$l-1$层的映射（输入$x$→线性层$l$的输入$x_l$），参数为$\theta_{\sim l}$（前$l-1$层参数） | 连接原始数据与线性层输入的“黑箱”，需满足Lipschitz连续                   |
| $c_l$             | 常数          | 前$l-1$层的$c_l$-Lipschitz连续常数                                                       | 衡量前层“输入变化→输出变化”的放大程度（值越小，前层越稳定）               |
| $G_l$             | 常数          | 损失函数对$\theta_{\sim l}$的梯度$\ell_2$范数上界（$\|\nabla_{\theta_{\sim l}} L\|_2 \leq G_l$） | 限制参数更新的最大幅度（避免梯度爆炸）                                   |
| $\Gamma_l$        | 常数          | 损失对“$\theta_{l,k}^{t-1}x_{n,l}^t$”的梯度$\ell_2$范数上界                               | 控制任务向量与输入关联的梯度稳定性                                       |
| $\Phi_l$          | 误差常数      | $\Phi_l = N \cdot c_l \cdot G_l \cdot \Gamma_l$（$N$为样本数）                             | 量化任务向量与输入加权和的误差上限                                       |
| $\omega_{i,l}^1, \omega_{i,l}^2$ | 重构常数 | 输入由任务向量重构时的固定常数（与重构系数、误差相关）                                       | Theorem 1中干扰上界的系数，由数据分布决定                               |
| $\eta_t$          | 学习率        | 微调第$t$次迭代的学习率                                                                     | 控制单次参数更新幅度（文档中设为1e-5）                                   |
| $\|\cdot\|_2$      | 范数          | $\ell_2$范数（欧几里得范数），$\|v\|_2 = \sqrt{\sum v_j^2}$                                | 衡量向量的“长度”（如输入变化幅度、误差大小）                             |
| $\|\cdot\|_F$      | 范数          | Frobenius范数，$\|M\|_F = \sqrt{\sum_{i,j} M_{i,j}^2}$                                      | 衡量矩阵的“整体大小”（如干扰向量与任务向量乘积的规模）                   |


## 二、理论推导核心链条
WUDI-Merging的理论基础是**“输入一致性→任务向量线性子空间→干扰上界”** 的三层推导，每层均以前一层为前提，最终支撑优化目标设计。


### 1. 基础：Lemma 1（输入一致性）—— 线性层输入迭代间稳定
#### 1.1 推导前提（符合实际微调场景）
文档明确4个关键假设（出处：3.2节），是推导的必要条件：
1. **小学习率**：微调学习率$\eta_t$极小（如1e-5），避免参数更新幅度过大；
2. **有限迭代**：微调迭代次数$T$有限（如300次），参数累积变化可控；
3. **Lipschitz连续**：前$l-1$层满足$c_l$-Lipschitz连续（输入微小变化→输出微小变化）；
4. **梯度有界**：损失对前$l-1$层参数的梯度$\ell_2$范数≤$G_l$（无梯度爆炸）。

#### 1.2 核心公式与推导
Lemma 1目标：证明“线性层输入在不同迭代间高度一致”，即$x_l^p$（迭代$p$）与$x_l^q$（迭代$q<p$）的差异有上界（出处：A.1.1节）。

**步骤1：定义前层映射与参数更新**  
前$l-1$层将原始输入$x$映射为线性层$l$的输入，即：  
$ x_l^t = f_{\sim l}(x; \theta_{\sim l}^t) \tag{1} $  
其中$\theta_{\sim l}^t$是前$l-1$层在迭代$t$的参数，更新规则为梯度下降：  
$ \theta_{\sim l}^{t+1} = \theta_{\sim l}^t - \eta_{t+1} \cdot \nabla_{\theta_{\sim l}^t} L(\theta^t) \tag{2} $  

**步骤2：计算参数累积变化**  
从迭代$q$到$p$，前$l-1$层参数的总变化为：  
$ \theta_{\sim l}^p - \theta_{\sim l}^q = - \sum_{i=q+1}^p \eta_i \cdot \nabla_{\theta_{\sim l}^{i-1}} L(\theta^{i-1}) \tag{3} $  
由“梯度有界”假设（$\|\nabla_{\theta_{\sim l}^{i-1}} L\|_2 \leq G_l$），结合$\ell_2$范数三角不等式：  
$ \|\theta_{\sim l}^p - \theta_{\sim l}^q\|_2 \leq \sum_{i=q+1}^p \eta_i \cdot \|\nabla_{\theta_{\sim l}^{i-1}} L\|_2 \leq G_l \cdot \sum_{i=q+1}^p \eta_i \tag{4} $  

**步骤3：关联参数变化与输入变化**  
由“Lipschitz连续”假设（$\|f_{\sim l}(x;\theta_1) - f_{\sim l}(x;\theta_2)\|_2 \leq c_l \cdot \|\theta_1 - \theta_2\|_2$），将$\theta_1=\theta_{\sim l}^p$、$\theta_2=\theta_{\sim l}^q$代入式(1)：  
$ \|x_l^p - x_l^q\|_2 = \|f_{\sim l}(x;\theta_{\sim l}^p) - f_{\sim l}(x;\theta_{\sim l}^q)\|_2 \leq c_l \cdot \|\theta_{\sim l}^p - \theta_{\sim l}^q\|_2 \tag{5} $  

**步骤4：最终输入一致性公式**  
将式(4)代入式(5)，得到Lemma 1的核心结论：  
$ \|x_l^p - x_l^q\|_2 \leq c_l \cdot G_l \cdot \sum_{i=q+1}^p \eta_i \tag{6} $  

#### 1.3 物理意义
当$\eta_t$小（如1e-5）且迭代次数少（如300次）时，$\sum_{i=q+1}^p \eta_i$极小（如3e-3），导致$\|x_l^p - x_l^q\|_2$接近0——即**线性层输入在迭代间几乎不变**，为后续任务向量与输入的关联奠定基础。


### 2. 核心：Proposition 1（任务向量的近似线性组合）—— 任务向量≈输入加权和
#### 2.1 推导前提
基于Lemma 1的“输入一致性”，以及“任务向量是梯度累积结果”的定义（$\tau_i = \sum_{t=1}^T -\eta_t \cdot \nabla_{\theta^{t-1}} L$）。

#### 2.2 核心公式与推导
Proposition 1目标：证明“线性层任务向量可近似表示为输入样本的加权和”（出处：A.1.2节）。

**步骤1：任务向量的梯度累积分解**  
线性层$l$第$k$个神经元的任务向量$\tau_{l}^k$，是微调各迭代参数更新的总和。通过链式法则，将参数梯度拆分为“损失对参数-输入乘积的梯度”与“输入向量”的乘积：  
$ \tau_{l}^k = \sum_{t=1}^T -\eta_t \cdot \nabla_{\theta_{l,k}^{t-1}} L(\theta^{t-1}) = \sum_{t=1}^T -\eta_t \sum_{n=1}^N \frac{\partial L(\theta^{t-1})}{\partial(\theta_{l,k}^{t-1} x_{n,l}^{t-1})} \cdot (x_{n,l}^{t-1})^\top \tag{7} $  
其中$x_{n,l}^{t-1}$是迭代$t-1$时第$n$个样本在层$l$的输入，$\frac{\partial L}{\partial(\theta_{l,k}^{t-1} x_{n,l}^{t-1})}$是损失对“参数-输入乘积”的梯度（记为$\gamma_{n,l,k}^t$）。

**步骤2：利用输入一致性简化输入项**  
由Lemma 1，$x_{n,l}^{t-1}$（迭代$t-1$）与$x_{n,l}^T$（最终迭代$T$）高度一致，即：  
$ x_{n,l}^{t-1} = x_{n,l}^T + \Delta x_{n,l}^{t-1} \quad (\Delta x_{n,l}^{t-1} \text{ 极小}) \tag{8} $  
将式(8)代入式(7)，拆分任务向量为“输入加权和”与“误差项”：  
$ \tau_{l}^k = \sum_{n=1}^N \underbrace{\left( \sum_{t=1}^T -\eta_t \gamma_{n,l,k}^t \right)}_{\beta_{n,l}^k} \cdot (x_{n,l}^T)^\top + \text{误差项} \tag{9} $  
其中$\beta_{n,l}^k$是第$n$个样本对$\tau_{l}^k$的贡献权重（文档定义的系数）。

**步骤3：误差项的上界估计**  
对式(9)取$\ell_2$范数，结合Lemma 1的输入差异上界（$\|x_{n,l}^{t-1} - x_{n,l}^T\|_2 \leq c_l G_l \sum_{i=t}^T \eta_i$）与“梯度$\gamma_{n,l,k}^t \leq \Gamma_l$”假设，通过三角不等式放缩：  
$ \left\| \tau_{l}^k - \sum_{n=1}^N \beta_{n,l}^k (x_{n,l}^T)^\top \right\|_2 \leq \underbrace{N \cdot c_l \cdot G_l \cdot \Gamma_l}_{\Phi_l} \cdot \sum_{t=1}^T \sum_{i=t}^T \eta_t \eta_i \tag{10} $  

#### 2.3 物理意义
式(10)右侧是误差上界：当$\eta_t$小且迭代少，右侧极小→$\tau_{l}^k \approx \sum_{n=1}^N \beta_{n,l}^k (x_{n,l}^T)^\top$，即**线性层任务向量构成输入的近似线性子空间**。这意味着：任务向量本身包含输入的核心信息，无需直接使用数据即可间接利用输入信息——解决“无数据融合”的核心痛点。


### 3. 关键：Theorem 1（干扰上界）—— 干扰可通过任务向量控制
#### 3.1 干扰的定义
融合模型对任务$i$的干扰，是“输入经过融合模型与专家模型的输出误差期望”（出处：3.1节），公式推导如下：  
$ \mathcal{J}_i(\tau_{m,l}) = \mathbb{E}_{x_{i,l} \sim p(x_{i,l})} \|\theta_{m,l} x_{i,l} - \theta_{i,l} x_{i,l}\|_2^2 \tag{11} $  
由$\theta_{m,l} = \theta_l + \tau_{m,l}$、$\theta_{i,l} = \theta_l + \tau_{i,l}$，代入得：  
$ \mathcal{J}_i(\tau_{m,l}) = \mathbb{E}_{x_{i,l}} \|(\tau_{m,l} - \tau_{i,l}) x_{i,l}\|_2^2 = \mathbb{E}_{x_{i,l}} \|\delta_{i,l} x_{i,l}\|_2^2 \tag{12} $  
其中$\delta_{i,l} = \tau_{m,l} - \tau_{i,l}$（干扰向量），式(12)表明：**干扰本质是干扰向量作用于输入的误差期望**。

#### 3.2 Theorem 1的核心公式与推导
Theorem 1目标：为$\mathcal{J}_i(\tau_{m,l})$提供上界，将干扰与任务向量关联（出处：A.1.3节）。

**步骤1：输入重构（基于Proposition 1）**  
由Proposition 1，输入$x_{i,l}$可由任务向量$\tau_{i,l}$的线性组合重构（加重构误差$\varepsilon(x_{i,l})$）：  
$ x_{i,l} = \sum_{k=1}^K \alpha_{i,l}^k(x_{i,l}) (\tau_{i,l}^k)^\top + \varepsilon(x_{i,l}) \tag{13} $  
其中$\alpha_{i,l}^k(x_{i,l})$是重构系数，$\tau_{i,l}^k$是$\tau_{i,l}$的第$k$个神经元，$\varepsilon(x_{i,l})$是重构误差。

**步骤2：代入干扰公式并放缩**  
将式(13)代入式(12)，干扰变为：  
$ \mathcal{J}_i(\tau_{m,l}) = \mathbb{E}_{x_{i,l}} \left\| \delta_{i,l} \left( \sum_{k=1}^K \alpha_{i,l}^k (\tau_{i,l}^k)^\top + \varepsilon(x_{i,l}) \right) \right\|_2^2 \tag{14} $  
利用**三角不等式**（$\|a + b\|_2 \leq \|a\|_2 + \|b\|_2$）和**Cauchy-Schwarz不等式**（$(\sum a_k b_k)^2 \leq (\sum a_k^2)(\sum b_k^2)$），对式(14)放缩：  
$ \mathcal{J}_i(\tau_{m,l}) \leq \mathbb{E}_{x_{i,l}} \left[ \left( \sum_{k=1}^K |\alpha_{i,l}^k| \cdot \|\delta_{i,l} (\tau_{i,l}^k)^\top\|_2 + \|\delta_{i,l} \varepsilon(x_{i,l})\|_2 \right)^2 \right] \tag{15} $  

**步骤3：引入重构常数并简化**  
定义重构常数：  

- $\omega_{i,l}^1 = \mathbb{E}_{x_{i,l}} \left( \sum_{k=1}^K |\alpha_{i,l}^k|^2 + 1 \right)$（与重构系数相关）；  
- $\omega_{i,l}^2 = \mathbb{E}_{x_{i,l}} \left( (\sum_{k=1}^K |\alpha_{i,l}^k|^2 + 1) \cdot \|\varepsilon(x_{i,l})\|_2^2 \right)$（与重构误差相关）。  

结合$\sum_{k=1}^K \|\delta_{i,l} (\tau_{i,l}^k)^\top\|_2^2 = \|\delta_{i,l} \tau_{i,l}^\top\|_F^2$（F范数定义），最终得到Theorem 1的干扰上界：  
$ \mathbb{E}_{x_{i,l}} \|\delta_{i,l} x_{i,l}\|_2^2 \leq \omega_{i,l}^1 \cdot \|\delta_{i,l} \tau_{i,l}^\top\|_F^2 + \omega_{i,l}^2 \cdot \|\delta_{i,l}\|_F^2 \tag{16} $  

#### 3.3 物理意义
式(16)将“干扰”（左侧）转化为“干扰向量与任务向量的相互作用”（右侧第一项）和“干扰向量自身规模”（右侧第二项）——这意味着：**最小化右侧即可间接最小化干扰**，为WUDI-Merging的优化目标提供理论依据。


## 三、WUDI-Merging方法的核心公式与优化逻辑
基于上述理论推导，WUDI-Merging通过“最小化干扰上界”设计优化目标，最终生成融合任务向量。


### 1. 优化目标设计
#### 1.1 原始优化目标（基于Theorem 1）
Theorem 1的上界中，$\omega_{i,l}^1, \omega_{i,l}^2$依赖输入数据（不可得），文档用经验系数$\omega$替代，并引入**任务权重平衡**（除以$\|\tau_{i,l}\|_F^2$，因大任务向量对干扰更敏感），得到初始优化目标（出处：3.3节）：  
$ \min_{\delta_{i,l}} \sum_{i=1}^P \frac{1}{\|\tau_{i,l}\|_F^2} \left( \|\delta_{i,l} \tau_{i,l}^\top\|_F^2 + \omega \cdot \|\delta_{i,l}\|_F^2 \right) \tag{17} $  

#### 1.2 转化为融合向量的优化
由$\delta_{i,l} = \tau_{m,l} - \tau_{i,l}$，将优化变量从$\delta_{i,l}$替换为$\tau_{m,l}$，式(17)等价于：  
$ \min_{\tau_{m,l}} \sum_{i=1}^P \frac{1}{\|\tau_{i,l}\|_F^2} \left( \|(\tau_{m,l} - \tau_{i,l}) \tau_{i,l}^\top\|_F^2 + \omega \cdot \|\tau_{m,l} - \tau_{i,l}\|_F^2 \right) \tag{18} $  

#### 1.3 简化优化目标（移除正则化）
文档通过prior work（Smith et al., 2021）证明：**梯度下降（如Adam）可引入隐式正则化**，无需显式$\omega$。因此移除正则化项，得到最终优化目标（算法1中损失函数的理论形式）：  
$ \min_{\tau_{m,l}} L_l = \sum_{i=1}^P \frac{1}{\|\tau_{i,l}\|_F^2} \cdot \|(\tau_{m,l} - \tau_{i,l}) \tau_{i,l}^\top\|_F^2 \tag{19} $  


### 2. 融合向量的求解：梯度下降与闭解
#### 2.1 梯度下降求解（算法1核心）
文档采用Adam优化器（学习率$\zeta=1e-5$，迭代$N=300$次），对每层$l$独立更新$\tau_{m,l}$。损失函数$L_l$对$\tau_{m,l}$的梯度为（出处：A.1.4节）：  
$ \nabla_{\tau_{m,l}} L_l = \sum_{i=1}^P \frac{2}{\|\tau_{i,l}\|_F^2} \cdot (\tau_{m,l} \tau_{i,l}^\top - \tau_{i,l} \tau_{i,l}^\top) \tau_{i,l} \tag{20} $  
更新规则为（对应算法1第9行）：  
$ \tau_{m,l}^{(n)} = \tau_{m,l}^{(n-1)} - \zeta \cdot \nabla_{\tau_{m,l}^{(n-1)}} L_l \tag{21} $  
其中$\tau_{m,l}^{(n)}$是第$n$次迭代的融合向量。

#### 2.2 闭解形式（WUDI-Merging-CFS）
当GPU资源不足时，可通过“梯度为零”求解式(18)的闭解（出处：A.1.4节）。令$\nabla_{\tau_{m,l}} L_l = 0$，整理得矩阵方程：  
$ \tau_{m,l} \cdot A = B \tag{22} $  
其中：  
- $A = \sum_{i=1}^P \frac{1}{\|\tau_{i,l}\|_F^2} (\tau_{i,l}^\top \tau_{i,l} + \omega I)$（系数矩阵，$I$为单位矩阵）；  
- $B = \sum_{i=1}^P \frac{1}{\|\tau_{i,l}\|_F^2} \tau_{i,l} (\tau_{i,l}^\top \tau_{i,l} + \omega I)$（常数矩阵）。  

若$A$可逆，闭解为：  
$ \tau_{m,l} = B \cdot A^{-1} \tag{23} $  
文档实验表明：闭解对$\omega$敏感，性能弱于梯度下降（隐式正则化更优）。


### 3. 最终融合模型参数
所有线性层的$\tau_{m,l}$优化完成后，组装为整体融合任务向量$\tau_m = \{\tau_{m,l}\}_{l=1}^\Psi$（$\Psi$为线性层总数），最终融合模型参数为（对应算法1第15行）：  
$ \theta_m = \theta + \tau_m \tag{24} $  


## 四、理论与方法的核心关联
| 理论模块                | 核心贡献                                  | 支撑的方法环节                          | 关键公式链接               |
|-------------------------|-------------------------------------------|-----------------------------------------|----------------------------|
| Lemma 1（输入一致性）   | 证明输入迭代间稳定                        | Proposition 1的输入简化                 | 式(6)→式(8)                |
| Proposition 1（线性子空间） | 任务向量≈输入加权和，间接利用数据信息      | Theorem 1的输入重构                     | 式(10)→式(13)              |
| Theorem 1（干扰上界）   | 干扰与任务向量关联，提供优化方向          | 优化目标设计（最小化干扰上界）          | 式(16)→式(18)              |
| 梯度下降优化            | 隐式正则化，避免调参                      | 算法1的逐层更新                        | 式(20)→式(21)              |


## 总结
WUDI-Merging的本质是**“以理论推导驱动方法设计”**：从“输入一致性”（Lemma 1）出发，证明“任务向量可代表输入”（Proposition 1），进而将“干扰”转化为“任务向量与干扰向量的作用”（Theorem 1），最终通过梯度下降最小化该作用，实现无数据、低干扰的模型融合。所有公式与符号均服务于这一逻辑，且严格遵循“无数据、低资源”的约束，实验中10.9%的性能提升也验证了理论推导的有效性。