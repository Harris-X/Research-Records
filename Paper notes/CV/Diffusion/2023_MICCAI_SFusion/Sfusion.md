[cite_start]根据该研究论文，SFusion（Self-attention Based N-to-One Multimodal Fusion Block）的核心贡献在于解决了多模态学习中常见的模态缺失问题，即 N-to-One 融合问题，其中输入模态的数量（N）是可变的。 [cite: 16, 17]

### 核心贡献

该论文的主要贡献有以下三点：
* [cite_start]**提出了一种与数据相关的自适应融合策略**：SFusion 是一种基于学习的融合策略，它能够自动学习不同模态之间的潜在相关性，并自适应地构建一个共享的特征表示。 [cite: 74, 75] [cite_start]关键在于，这个过程不需要通过填充零值或合成数据来“伪装”缺失的模态。 [cite: 18, 74]
* [cite_start]**具有通用性和可集成性**：SFusion 的设计使其能够轻松集成到现有的各种多模态分析网络中。 [cite: 21, 76] [cite_start]它可以接收来自任何上游处理模型的输入，并将其输出提供给下游的决策模型，因此适用于不同的任务和骨干网络。 [cite: 77, 283]
* [cite_start]**经过充分的实验验证**：研究者在人类活动识别（使用 SHL 数据集）和脑肿瘤分割（使用 BraTS2020 数据集）两个任务上进行了广泛的实验。 [cite: 22, 78] [cite_start]实验结果表明，与现有的其他融合策略相比，SFusion 取得了更优的性能。 [cite: 23, 79]

### 方法介绍

[cite_start]SFusion 的目标是学习一个融合函数 $F$，该函数能将一组来自所有可用模态的特征表示 $I$ 映射到一个共享的特征表示 $f_s$，表示为 $F(I) \rightarrow f_s$。 [cite: 86] [cite_start]该模型的架构主要由两个模块组成：相关性提取（Correlation Extraction, CE）模块和模态注意力（Modal Attention, MA）模块。 [cite: 69, 88]

**输入与符号定义**:
* [cite_start]$K$: 当前可用的模态集合，是所有可能模态总集 $S$ 的一个子集。 [cite: 82]
* [cite_start]$f_k$: 第 $k$ 个模态的输入特征表示，其维度为 $\mathbb{R}^{B \times C \times R_f}$。 [cite: 83]
* [cite_start]$B$: 批量大小（batch size）。 [cite: 83]
* [cite_start]$C$: 通道数（number of channels）。 [cite: 83]
* [cite_start]$R_f$: 特征表示的形状，可以是一维（$L$）、二维（$H \times W$）或三维（$D \times H \times W$）等。 [cite: 84]
* [cite_start]$I$: 所有可用模态的特征表示集合，即 $I = \{f_k | k \in K\}$。 [cite: 85]

#### 1. 相关性提取 (Correlation Extraction, CE) 模块

[cite_start]CE 模块的目标是学习不同可用模态之间的潜在相关性。 [cite: 70, 125]
1.  [cite_start]**令牌化 (Tokenization)**: 首先，将每个可用模态的特征图 $f_k$ 的 $R_f$ 维度展平，然后将所有模态的特征连接起来，形成一个新的张量 $z_0 \in \mathbb{R}^{B \times T \times C}$。 [cite: 123, 124] [cite_start]这里的 $T=R \times |K|$，$|K|$ 是可用模态的数量。 [cite: 124] [cite_start]这个过程将输入的特征图转换为了令牌（tokens）。 [cite: 19]
2.  [cite_start]**自注意力层 (Self-Attention Layers)**: 随后，$z_0$ 被送入一个由多个自注意力层（SAL）组成的堆栈中。 [cite: 125] [cite_start]每个 SAL 包含一个多头注意力（MHA）块和一个全连接前馈网络（FFN），并在每个块之前应用层归一化（LN）。 [cite: 126, 127] 第 $x$ 层的计算过程如下：
    [cite_start]$$z_{x}^{\prime} = MHA(LN(z_{x-1})) + z_{x-1}$$ [cite: 128]
    [cite_start]$$z_{x} = FFN(LN(z_{x}^{\prime})) + z_{x}^{\prime}$$ [cite: 129]
3.  [cite_start]**输出**: 经过所有 SAL 层后得到的最终输出 $z_l$，通过重塑（reshape）和分割（split）操作，被还原成与输入格式相同的特征表示集合 $I' = \{f'_k | k \in K\}$。 [cite: 132, 134, 136] [cite_start]这个新的集合 $I'$ 包含了学习到的多模态相关性信息。 [cite: 136]

#### 2. 模态注意力 (Modal Attention, MA) 模块

[cite_start]MA 模块接收来自 CE 模块的输出 $I'$，并利用它来生成权重图，最终融合原始输入特征。 [cite: 71, 138]
1.  [cite_start]**权重图生成**: 该模块引入了一种模态级和体素级的 softmax 函数，根据 $I'$ 生成每个模态的权重图 $m_k$。 [cite: 142] 权重图的计算公式如下：
    [cite_start]$$m_{k}^{i} = \frac{e^{v_{k}^{i}}}{\sum_{j\in K}e^{v_{j}^{i}}}$$ [cite: 145]
    * [cite_start]$m_k^i$ 是第 $k$ 个模态的权重图中第 $i$ 个体素（voxel）的值。 [cite: 143]
    * [cite_start]$v_k^i$ 是从 CE 模块输出的特征 $f'_k$ 中对应的第 $i$ 个体素的值。 [cite: 143]
    * [cite_start]$e$ 是自然对数。 [cite: 143]
    * [cite_start]$K$ 是所有可用模态的集合。 [cite: 145]

2.  [cite_start]**特征融合**: 最后，将原始的输入特征图 $f_k$ 与其对应的权重图 $m_k$ 进行元素级相乘，然后将所有可用模态的结果相加，得到最终的融合特征图 $f_s$。 [cite: 147, 148]
    [cite_start]$$f_s = \sum_{k \in K} f_k \cdot m_k$$ [cite: 148]

[cite_start]这种设计确保了融合特征 $f_s$ 的值范围保持稳定，并且保留了 CE 模块学习到的模态间相对重要性。 [cite: 150, 151] [cite_start]一个特殊情况是，当只有一个模态可用时（$|K|=1$），其权重图所有值都为 1，这意味着融合结果就是其自身（$f_s = f_k$），从而保持了原始特征信息。 [cite: 152, 153]