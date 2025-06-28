# 《IterIS 方法：解决 LoRA 合并问题的有效途径》

好的，我们来详细解读这篇名为《IterIS: Iterative Inference-Solving Alignment for LORA Merging》的论文中提出的方法。

### 核心目标：优化LoRA合并

[cite_start]首先，我们要理解论文的目标。大型预训练模型（如GPT、Stable Diffusion）可以通过一种名为“低秩适应”（Low-Rank Adaptation, LoRA）的技术进行微调，以适应特定的下游任务（如生成特定风格的图片、执行特定的NLP任务）[cite: 5]。这样，对于每个任务，我们都会得到一个轻量级的LoRA“适配器”。

[cite_start]**LoRA合并**技术旨在将多个任务的LoRA适配器融合成一个**统一的适配器 (Unified Adapter)** [cite: 7, 77][cite_start]。这样做的好处是，我们只需要一个模型就能处理多种任务，同时避免了在混合数据上进行资源密集型的重新训练，并能保护数据隐私 [cite: 7, 34]。

[cite_start]IterIS是作者提出的一种新颖的、基于优化的LoRA合并算法，旨在解决现有方法的几大局限性 [cite: 9]。

### 现有方法的局限性

在深入IterIS之前，我们需要先了解它所要改进的两种主流合并方法。

#### 1. 线性合并 (Linear Merging)

[cite_start]这是最简单的方法，直接将多个LoRA的权重进行线性组合（通常是平均）[cite: 99]。

* **公式**:
    [cite_start]$W^{*} = \frac{1}{N} (W_1 + \cdots + W_N)$ [cite: 90]
* **符号含义**:
    * $W^*$：代表合并后得到的**统一适配器**的权重矩阵。
    * $W_i$：代表第 $i$ 个特定任务的LoRA权重矩阵。
    * $N$：代表被合并的LoRA总数。
* [cite_start]**问题**: 这种方法基于一个“各向同性假设”（isotropic assumption），即假定输入到LoRA的特征遵循简单的各向同性分布 [cite: 39, 105][cite_start]。然而在实际应用中，这个假设通常不成立，导致模型性能严重下降 [cite: 40, 105]。

#### 2. 基于真实分布的合并 (Real-distribution-based Merging)

[cite_start]为了克服线性合并的缺点，后续方法尝试利用从真实数据中提取的特征来指导合并过程 [cite: 108][cite_start]。其核心思想是让统一适配器的输出尽可能地与每个单独LoRA的输出对齐 [cite: 22, 44]。

* **公式**:
    [cite_start]$W^{*} = (\sum_{i=1}^{N} X_{i}X_{i}^{T})^{-1} (\sum_{i=1}^{N} X_{i}X_{i}^{T}W_{i})$ [cite: 112]
* **符号含义**:
    * $W^*$ 和 $W_i$ 的含义同上。
    * [cite_start]$X_i$：代表输入到第 $i$ 个**特定任务LoRA**的特征矩阵。这些特征是通过在一些无标签样本上进行推理得到的 [cite: 109, 111]。
* **作者指出的该方法的三个主要问题**:
    1.  [cite_start]**粗略假设 (Rough Assumption)**: 该方法假设输入到**统一适配器**的特征与输入到**各个独立LoRA**的特征是相同的（即公式中求解 $W^*$ 时，其输入被近似为 $X_i$） [cite: 46, 113][cite_start]。然而，随着模型深度的增加，这种差异会越来越大，从而限制了性能 [cite: 46, 117]。
    2.  [cite_start]**大量样本需求 (Massive Sample Requirement)**: 为了保证公式中矩阵 $(\sum_{i=1}^{N} X_{i}X_{i}^{T})$ 的可逆性和算法的鲁棒性，通常需要大量的无标签样本 [cite: 48, 119]。
    3.  [cite_start]**优化不平衡 (Unbalanced Optimization)**: 如果某个任务的输入特征 $X_i$ 的数值幅度远大于其他任务，它将在优化过程中占据主导地位，导致最终模型在其他任务上表现不佳 [cite: 49, 121]。

---

### IterIS算法详解

IterIS通过三个关键改进来解决上述问题。其核心公式如下：

* **核心优化公式 (IterIS)**:
    [cite_start]$W^{*} = (\sum_{i=1}^{N} \lambda_{i} \tilde{X}_{i}\tilde{X}_{i}^{T})^{-1} (\sum_{i=1}^{N} \lambda_{i} \tilde{X}_{i}X_{i}^{T}W_{i})$ [cite: 133]

我们来逐一解析其改进之处和符号含义。

#### 1. 改进一：迭代推理-求解 (Iterative Inference-Solving)

这项改进直接针对**“粗略假设”**问题。IterIS不再假设统一适配器的输入特征与独立LoRA的输入特征相同，而是通过迭代的方式动态地更新和优化它。

* **关键符号**:
    * $X_i$: 含义不变，是输入到**原始独立LoRA**的特征矩阵。
    * [cite_start]$\tilde{X}_i$: 这是IterIS引入的关键变量，代表输入到**统一适配器 $W^*$** 的特征矩阵 [cite: 131, 135]。

* [cite_start]**迭代过程** (见论文图4(c)和算法1)[cite: 94, 115]:
    1.  [cite_start]**初始化**: 在第一次迭代时，我们没有统一适配器，因此假设 $\tilde{X}_i = X_i$ [cite: 133]。
    2.  [cite_start]**求解步骤 (Solving Step)**: 将当前的 $X_i$ 和 $\tilde{X}_i$ 代入IterIS的核心公式，计算出一个版本的统一适配器 $W^*$ [cite: 136]。
    3.  [cite_start]**推理步骤 (Inference Step)**: 将上一步得到的 $W^*$ 应用到预训练模型上，构成一个新的“统一模型”。然后，再次将无标签样本输入这个统一模型进行推理，提取出位于LoRA位置的**新的输入特征**，这就是下一次迭代用的 $\tilde{X}_i$ [cite: 134]。
    4.  [cite_start]**循环**: 重复步骤2和3，直到收敛或达到最大迭代次数 [cite: 126]。

[cite_start]通过这种方式，IterIS逐步精确化了对统一适配器输入特征 $\tilde{X}_i$ 的估计，从而打破了“粗略假设”，得到了更准确的合并结果 [cite: 138, 139]。

#### 2. 改进二：高效正则化 (Few-Sample Requirement)

[cite_start]这项改进针对**“大量样本需求”**问题。IterIS引入了一个正则化项，使其在仅有少量样本（约占之前方法的1-5%）的情况下也能稳健运行 [cite: 11, 140]。

* [cite_start]**公式**: 在计算中，将矩阵 $A$ 替换为 $A + \alpha ||A||_F I$ [cite: 142]。
    * 具体来说，$\tilde{X}_{i}\tilde{X}_{i}^{T}$ 被替换为 $\tilde{X}_{i}\tilde{X}_{i}^{T}+\alpha||\tilde{X}_{i}\tilde{X}_{i}^{T}||_{F}I$。
    * $\tilde{X}_{i}X_{i}^{T}$ 被替换为 $\tilde{X}_{i}X_{i}^{T}+\alpha||\tilde{X}_{i}X_{i}^{T}|_{F}I$。
* **符号含义**:
    * [cite_start]$\alpha$: 一个很小的正则化系数（如 $1 \times 10^{-4}$）[cite: 144]。
    * $||\cdot||_F$: 弗罗贝尼乌斯范数 (Frobenius norm)，衡量矩阵的大小。
    * [cite_start]$I$: 单位矩阵 (identity matrix) [cite: 142]。
* [cite_start]**作用**: 这个正则项通过给矩阵的对角线增加一个小的数值，确保了矩阵的可逆性，同时减轻了在少量样本上的过拟合风险，增强了算法的鲁棒性 [cite: 143]。

#### 3. 改进三：自适应权重 (Adaptive Weight Balancing)

这项改进针对**“优化不平衡”**问题。IterIS为每个任务引入了自适应的权重 $\lambda_i$，以平衡不同任务在优化过程中的影响。

* **公式**:
    [cite_start]$\lambda_{i} = ||W_{i}||_{F}^{2} \cdot ||W_{i}^{T}X_{i}||_{F}^{-2}$ [cite: 146]
* **符号含义**:
    * $||W_i||_F^2$: 第 $i$ 个LoRA权重矩阵大小的平方。
    * $||W_i^T X_i||_F^{-2}$: 第 $i$ 个LoRA**输出**特征矩阵大小平方的倒数。
* [cite_start]**作用**: 这个权重 $\lambda_i$ 的设计非常巧妙。如果一个任务的输出特征 $W_i^T X_i$ 的幅度（范数）很大，那么它的权重 $\lambda_i$ 就会相应变小。这相当于一个归一化过程，防止了某些任务因其输出数值较大而过度影响最终的合并结果，从而实现了更平衡的优化 [cite: 146]。

### 总结

总而言之，IterIS方法通过一个**迭代推理-求解框架**来获得更精确的合并目标，通过**正则化项**来减少对样本数量的依赖，并通过**自适应权重**来平衡不同任务的贡献。这使得它能够比先前的方法更高效、更鲁棒地将多个LoRA模块融合成一个功能强大的统一适配器。







论文提出 IterIS 方法用于 LoRA 合并，以解决现有方法在特征假设、样本需求和优化平衡方面的问题，在多个领域实验中展现出优异性能。下面从 LoRA 合并的背景、现有方法的局限性以及 IterIS 算法的改进等方面，详细介绍论文的方法、公式、符号和概念。

**LoRA 合并的背景与相关概念**

**定义**：给定预训练模型和 N 个不同任务，得到 N 组特定任务的 LoRAs，它们在预训练模型的相同位置进行微调。将这些 LoRAs 合并成统一适配器，得到统一模型。

**线性合并**：最直接的 LoRA 合并方法，通过线性组合单个 LoRAs 的参数创建统一适配器。公式为$W^{*}=\underset{W}{arg min } \mathbb{E}_{\mathcal{X}}\left[\sum_{i = 1}^{N} \lambda_{i}\left\| W_{i}^{T} \mathcal{X}_{i}-W^{T} \mathcal{X}_{i}\right\| _{F}^{2}\right]$ ，其中$\mathbb{E}_{X}[\cdot]$是关于$x$的期望，$\lambda_{i}$是常数，$\|\cdot\|_{F}$是 Frobenius 范数。假设$x_{i}$服从各向同性分布且相互独立时，有闭式解$W^{*}=\sum_{i = 1}^{N} \overline{\lambda}_{i} W_{i}, \overline{\lambda}_{i}=\frac{\lambda_{i} \mathbb{E}_{\mathcal{X}_{i}}\left[\left\| \mathcal{X}_{i}\right\| _{F}^{2}\right]}{\sum_{j = 1}^{N} \lambda_{j} \mathbb{E}_{\mathcal{X}_{j}}\left[\left\| \mathcal{X}_{j}\right\| _{F}^{2}\right]}$ 。但该假设在实际中常不成立，会限制统一模型性能。

**基于真实分布的合并**：为放宽各向同性分布假设，利用真实分布的特征进行 LoRA 合并。通过对无标签样本进行推理获取 LoRAs 的输入特征，近似公式中的期望。优化问题可表示为$W^{*}=\underset{W}{arg min } \sum_{i = 1}^{N}\left\| W_{i}^{T} X_{i}-W^{T} X_{i}\right\| _{F}^{2}$ ，有闭式解$W^{*}=\left(\sum_{i = 1}^{N} X_{i} X_{i}^{T}\right)^{-1}\left(\sum_{i = 1}^{N} X_{i} X_{i}^{T} W_{i}\right)$ 。不过，该方法存在粗糙假设、大量样本需求和优化不平衡的问题。

**IterIS 算法**

**迭代推理求解**：基于真实分布合并方法的核心原理，用$\tilde{X}_{i}$表示统一适配器的输入特征，优化目标为$W^{*}=\underset{W}{arg min } \sum_{i = 1}^{N} \lambda_{i}\left\| W_{i}^{T} X_{i}-W^{T} \overline{X}_{i}\right\| _{F}^{2}$ ，闭式解为$W^{*}=\left(\sum_{i = 1}^{N} \lambda_{i} \overline{X}_{i} \overline{X}_{i}^{T}\right)^{-1}\left(\sum_{i = 1}^{N} \lambda_{i} \overline{X}_{i} X_{i}^{T} W_{i}\right)$ 。在每次迭代中，通过推理提取统一适配器的输入特征$\overline{X}_{i}$ ，更新优化目标并计算统一适配器的解，逐步细化优化目标以提高性能。

**少样本需求**：引入正则化项$\overline{X}_{i} \overline{X}_{i}^{T}+\alpha\left\| \overline{X}_{i} \overline{X}_{i}^{T}\right\| _{F} I, \overline{X}_{i} X_{i}^{T}+\alpha\left\| \overline{X}_{i} X_{i}^{T}\right\| _{F} I$ ，减少对大量无标签样本的依赖，仅需 1 - 5% 的无标签样本，还能缓解过拟合，增强算法鲁棒性，实验中$\alpha$通常设为$1\times10^{-4}$或更低。

**自适应权重平衡**：为解决优化过程中的不平衡问题，引入自适应权重$\lambda_{i}=\left\| W_{i}\right\| _{F}^{2}\left\| W_{i}^{T} X_{i}\right\| _{F}^{-2}$ ，避免某些项在优化中被过度优先考虑。

**算法收敛性分析**：利用深度学习模型的有向无环图结构，推导得出算法收敛的迭代次数上限。对于具有 J 层自注意力模块的编码器，算法在$J - 1$次迭代后收敛，实验中为防止过拟合，将迭代次数上限设为 20 。



该部分主要介绍了 LoRA 合并方法的相关背景、现有方法的局限性，进而引出了 IterIS 算法，具体内容如下：

**预备知识**

**LoRA 合并的定义**：给定一个预训练模型和 N 个不同的任务，会得到 N 组特定任务的 LoRAs。这些 LoRAs 在对应的数据集上进行微调，并应用于预训练模型的相同位置。Wi 代表第 i 个任务特定 LoRA 的权重，将这 N 个任务特定的 LoRAs 合并得到的统一适配器用 W 表示 。把所有统一适配器集成到预训练模型后，得到的模型就是 “统一模型”。LoRA 合并无需标记数据或基于梯度的训练。

**线性合并**：这是最直接的 LoRA 合并方法，通过线性组合单个 LoRAs 的参数来创建统一适配器。其优化公式为$W ^ { * } = \underset{W}{arg min } \mathbb{E}_{\mathcal{X}}\left[\sum _ { i = 1 } ^ { N } \lambda _ { i } \left\| W _ { i } ^ { T } \mathcal{X} _ { i } - W ^ { T } \mathcal{X} _ { i } \right\| _ { F } ^ { 2 }\right]$ ，其中$\mathbb{E}_{X}[\cdot]$表示关于 X 的期望，$\lambda _ { i }$是常数，$\|\cdot\|_{F}$代表 Frobenius 范数。在假设每个$X _ { i }$服从各向同性分布且$X _ { 1 }... X _ { N }$相互独立的情况下，该式有闭式解$W ^ { * } = \sum _ { i = 1 } ^ { N } \overline { \lambda } _ { i } W _ { i }, \overline { \lambda } _ { i } = \frac { \lambda _ { i } \mathbb{E} _ { \mathcal{X} _ { i } } [ \| \mathcal{X} _ { i } \| _ { F } ^ { 2 } ] } { \sum _ { j = 1 } ^ { N } \lambda _ { j } \mathbb{E} _ { \mathcal{X} _ { j } } [ \| \mathcal{X} _ { j } \| _ { F } ^ { 2 } ] }$ 。但实际中各向同性分布假设往往不成立，会限制统一模型性能。此外，将加权系数$\overline { \lambda } _ { i }$当作超参数会使超参数空间过大，手动调整不切实际，所以通常采用平均 N 个任务特定 LoRAs 的方式，即$\overline { \lambda } _ { i } = \frac{1}{N}$。

**基于真实分布的合并**：为放宽各向同性分布假设，部分 LoRA 合并方法利用真实分布的特征。这些方法对无标签样本进行推理获取 LoRAs 的输入特征，并用这些特征近似公式中的数学期望。将所有$\lambda _ { i }$设为相等，得到优化公式$W ^ { * } = \underset{W}{arg min } \sum _ { i = 1 } ^ { N } \left\| W _ { i } ^ { T } X _ { i } - W ^ { T } X _ { i } \right\| _ { F } ^ { 2 }$ ，该优化可看作矩阵空间中的线性回归，统一适配器有闭式解$W ^ { * } = ( \sum _ { i = 1 } ^ { N } X _ { i } X _ { i } ^ { T } ) ^ { - 1 } ( \sum _ { i = 1 } ^ { N } X _ { i } X _ { i } ^ { T } W _ { i } )$ 。然而，现有基于真实分布的合并方法存在三个关键局限：一是依赖粗糙假设，认为统一适配器的输入特征与$X _ { i }$相同，但随着模型深度增加，这种近似会产生较大误差，限制性能；二是为保证矩阵可逆和增强鲁棒性，需要大量样本，在实际应用中往往难以实现；三是由于$W _ { i }$的权重基于内积矩阵$X _ { i } X _ { i } ^ { T }$ ，$X _ { i }$大小的变化会导致解受最突出项影响过大，使优化不平衡。

**IterIS 算法**

**迭代推理求解**：基于真实分布合并方法的核心原理，用$\tilde{X}_{i}$表示统一适配器的输入特征（取代近似的$X _ { i }$），得到更精确的 LoRA 合并表示公式$W ^ { * } = \underset{W}{arg min } \sum _ { i = 1 } ^ { N } \lambda _ { i } \left\| W _ { i } ^ { T } X _ { i } - W ^ { T } \overline{X} _ { i } \right\| _ { F } ^ { 2 }$ ，其闭式解为$W ^ { * } = ( \sum _ { i = 1 } ^ { N } \lambda _ { i } \overline{X} _ { i } \overline{X} _ { i } ^ { T } ) ^ { - 1 } ( \sum _ { i = 1 } ^ { N } \lambda _ { i } \overline{X} _ { i } X _ { i } ^ { T } W _ { i } )$ 。$\tilde{X}_{i}$从$X _ { i }$开始迭代更新，每次迭代包括两个步骤：一是推理步骤，对当前统一模型中的第 i 个任务样本进行推理，提取$\tilde{X}_{i}$作为统一适配器的输入特征；二是求解步骤，更新$\tilde{X}_{i}$后计算统一适配器的解$W ^ { * }$，并将其融入预训练模型，得到下一次迭代的新统一模型。通过直接利用统一适配器的输入特征，IterIS 减轻了粗糙假设的影响，迭代优化目标可实现更精确的合并表示。

**少样本需求**：与以往需要大量无标签样本的方法不同，IterIS 通过在公式（6）的内积矩阵中引入正则化，仅需少量（1 - 5%）无标签样本。具体是将内积矩阵替换为$\overline{X} _ { i } \overline{X} _ { i } ^ { T } + \alpha \| \overline{X} _ { i } \overline{X} _ { i } ^ { T } \| _ { F } I$和$\overline{X} _ { i } X _ { i } ^ { T } + \alpha \| \overline{X} _ { i } X _ { i } ^ { T } \| _ { F } I$ ，其中 I 是单位矩阵。这种正则化可减轻对样本的过拟合，增强算法鲁棒性，实验中通常将$\alpha$设为$1\times10 ^ { - 4 }$或更低。

**自适应权重平衡**：在公式（5）中为$\lambda _ { i }$设置均匀权重会导致与以往方法相同的不平衡问题。因此，IterIS 引入自适应权重$\lambda _ { i } = \frac{\| W _ { i } \| _ { F } ^ { 2 }}{\| W _ { i } ^ { T } X _ { i } \| _ { F } ^ { 2 }}$ ，以平衡优化过程，避免某些项被过度优先考虑。

**IterIS 分析**：利用深度学习模型的有向无环图结构，推导得出算法收敛的迭代次数上限。对于具有 J 层自注意力模块且 Q、K、V 矩阵用 LoRA 微调的编码器，算法在 J - 1 次迭代后收敛。为防止过拟合，实验中将迭代次数上限设为 20 。尽管迭代次数增加，但 IterIS 收敛所需样本数比以往方法少得多，总体推理次数低，计算效率高。并且该方法通过闭式解和逐层更新，避免了基于梯度训练的计算需求。算法的效率分析和完整过程在附录中展示，具体步骤见算法 1。