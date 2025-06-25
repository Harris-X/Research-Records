# 《DF-Merge：一种高效的多任务模型合并方法》

### 一段话总结

本文提出**动态 Fisher 加权模型合并方法（DF-Merge）**，通过**贝叶斯优化**动态调整任务特定模型的缩放系数，并结合**Fisher 信息**评估参数重要性，统一了模型级缩放和参数级重要性整合两种策略。实验表明，DF-Merge 在不同规模模型和多样任务上显著优于现有基线，仅需少量迭代和验证数据即可接近多任务微调性能，为无需额外训练数据的多任务模型构建提供了高效解决方案。

### 思维导图



```mindmap
## **研究背景**
- 预训练模型微调产生大量任务特定模型，模型合并是构建多任务模型的高效方式
- 现有方法分两类：模型级缩放（如Task Arithmetic）和参数级重要性整合（如Fisher Merging），但性能落后于多任务微调
- 目标：统一两类策略，提出更优合并框架

## **方法框架：Dynamic Fisher-weighted Merging (DF-Merge)**
- **统一模型合并框架**  
  - 通用形式结合模型级系数（λ_i）和参数级重要性（Fisher信息矩阵C_θi）  
  - 现有方法（如平均法、TA、Fisher Merging）为特殊情况
- **动态Fisher加权合并**  
  - 系数λ_i调整任务向量τ_i = θ_i - θ_pre，形成θ_i(λ_i) = θ_pre + λ_iτ_i  
  - 基于θ_i(λ_i)动态计算Fisher信息diag(ˆF_θi(λ_i))，而非固定原始模型参数
- **贝叶斯优化系数**  
  - 高斯过程建模验证集准确率，通过Expected Improvement（EI）或UCB选择最优λ_i  
  - 高效搜索非 differentiable 指标，减少迭代次数

## **核心技术贡献**  
- 统一模型合并的两种主流策略，提升灵活性  
- 引入贝叶斯优化优化非 differentiable 指标  
- 动态整合Fisher信息，捕捉参数重要性随系数变化的特性

## **实验结果**  
- **主实验**  
  - T5-base平均准确率比最佳基线高4.48%，T5-large高1.73%  
  - 缩小与多任务微调差距：T5-base差距3.55%，T5-large 3.15%  
  - 平衡多任务性能，最大任务精度下降小于基线
- **消融实验**  
  - 移除Fisher信息导致T5-base下降1.34%，T5-large 1.07%  
  - 移除贝叶斯优化导致T5-base下降5.52%，T5-large 13.71%  
- **效率分析**  
  - 10次随机初始化+9次迭代即可接近最优（差距0.59%）  
  - 仅需5%验证数据即可接近全量数据性能

## **结论与展望**  
- DF-Merge为多任务模型合并提供高效训练-free方案  
- 限制：需同架构预训练模型，依赖验证数据，Fisher信息使用对角线近似  
- 未来方向：跨架构合并、无验证数据场景、非对角线Fisher信息建模
```

### 详细总结

#### 1. 研究背景与目标

**现状**：

预训练模型微调生成大量任务特定模型（如 Hugging Face 超百万模型），但独立微调模型难以处理多任务。

多任务学习需同时访问所有任务数据，实际中数据获取困难；模型合并通过参数空间组合构建多任务模型，无需额外训练数据。

**现有方法局限**：

**模型级缩放**（如 Task Arithmetic）：统一缩放任务向量，忽略参数重要性差异。

**参数级重要性整合**（如 Fisher Merging）：基于固定模型参数的 Fisher 信息，未动态调整系数。

**目标**：提出统一框架，结合两类方法优势，提升模型合并性能。

#### 2. 方法：Dynamic Fisher-weighted Merging (DF-Merge)

**统一模型合并框架**：

通用公式：

$ 
    \theta^{*} = \left(\sum_{i=1}^{M} \text{diag}(\hat{F}_{\theta_i(\lambda_i)})\right)^{-1} \left(M \sum_{i=1}^{M} \text{diag}(\hat{F}_{\theta_i(\lambda_i)}) \lambda_i \tau_i\right) + \theta_{\text{pre}}
     $

其中$\tau_i = \theta_i - \theta_{\text{pre}}$为任务向量，$\lambda_i$为缩放系数，$\text{diag}(\hat{F}_{\theta_i(\lambda_i)})$为动态计算的 Fisher 信息（基于缩放后参数$\theta_i(\lambda_i) = \theta_{\text{pre}} + \lambda_i \tau_i$）。

**特殊情况**：

平均法：$\lambda_i = 1/M$，Fisher 信息为单位矩阵。

Task Arithmetic（GTA）：Fisher 信息为单位矩阵，优化$\lambda_i$。

Fisher Merging：固定$\lambda_i = 1$，使用原始模型 Fisher 信息。

**贝叶斯优化系数**：

目标：最大化验证集平均准确率，优化非 differentiable 指标。

流程：

高斯过程建模系数 - 准确率关系，初始 10 次随机采样。

每次迭代通过 EI 或 UCB 选择新系数，计算合并模型并评估准确率。

迭代 50 次，系数范围 \[0,1]，每次使用 30 个验证样本计算 Fisher 信息。

#### 3. 实验设置

**模型与数据集**：

模型：T5-base（223M 参数）、T5-large（738M 参数）。

任务：6 个 NLP 任务（PAWS、QASC、QuaRTz、Story Cloze、WikiQA、Winogrande），覆盖问答、 paraphrase 识别等。

**基线**：

平均法、Fisher Merging、Task Arithmetic、DARE、TIES-Merging。

**评估指标**：任务准确率，对比多任务微调（Oracle）性能。

#### 4. 关键实验结果

**主实验对比（Table 2）**：



| 模型       | 方法            | 平均准确率 (%) | 对比最佳基线提升 (%)               | 与多任务微调差距 (%) |
| -------- | ------------- | --------- | -------------------------- | ------------ |
| T5-base  | DF-Merge (EI) | 78.14     | +4.48 (vs TIES-Merging)    | 3.55         |
| T5-large | DF-Merge (EI) | 83.59     | +1.73 (vs Task Arithmetic) | 3.15         |

DF-Merge 在多数任务中准确率最高，且平衡性能（最大任务精度下降：T5-base 8.09%，T5-large 4.75%，均小于基线）。

**消融实验（Table 3）**：

**移除 Fisher 信息**（仅 GTA + 贝叶斯优化）：平均准确率下降 1-1.34%（T5-base）、0.97-1.07%（T5-large），证明参数重要性评估的必要性。

**移除贝叶斯优化**（仅 Fisher Merging）：平均准确率大幅下降 5.52%（T5-base）、13.71%（T5-large），证明动态系数优化的关键作用。

**效率分析**：

**迭代次数**：10 次随机初始化 + 9 次迭代即可达到接近最优性能（差距 0.59%），后续迭代增益有限（图 4）。

**验证集大小**：仅需 5% 验证数据即可接近全量数据性能，30% 数据时稳定超越 Task Arithmetic（图 5）。

#### 5. 结论与意义

**核心贡献**：

统一模型合并的两类策略，提出更灵活的框架。

引入贝叶斯优化优化非 differentiable 指标，动态调整系数。

结合动态 Fisher 信息，捕捉参数重要性随系数变化的特性，提升多任务性能。

**价值**：

无需额外训练数据，高效构建多任务模型，缩小与多任务微调的性能差距。

低数据需求和少迭代次数，适合实际应用中快速合并现有模型。

### 关键问题与答案

#### 问题 1：DF-Merge 相比现有模型合并方法的核心创新点是什么？

**答案**：DF-Merge 的核心创新在于**统一模型级缩放和参数级重要性整合策略**，通过**贝叶斯优化动态调整任务向量的缩放系数**，并**基于缩放后的参数动态计算 Fisher 信息**以评估参数重要性。传统方法要么固定系数（如 Fisher Merging），要么忽略参数重要性动态变化（如 Task Arithmetic），而 DF-Merge 同时优化系数和整合动态 Fisher 信息，从而更高效地搜索低损失区域，提升多任务性能。

#### 问题 2：实验中 DF-Merge 的性能提升有多显著？能否举例说明？

**答案**：DF-Merge 在不同模型和任务上显著优于基线。例如，在 T5-base 模型上，其平均准确率为 78.14%，比次优基线 TIES-Merging（72.62%）提升 5.52%；在 T5-large 上，平均准确率 83.59%，超过 Task Arithmetic（81.86%）1.73%。与多任务微调相比，T5-base 差距从传统方法的 10% 以上缩小至 3.55%，T5-large 缩小至 3.15%，证明其接近有监督多任务学习的性能。

#### 问题 3：DF-Merge 在效率和数据需求上有何优势？

**答案**：DF-Merge 在效率和数据需求上表现优异：

**迭代次数**：仅需 10 次随机初始化 + 约 10 次贝叶斯优化迭代（共约 20 次）即可接近最优性能，而基线方法（如网格搜索）需数十次评估。

**验证数据量**：使用 5% 的验证数据时，性能已接近全量数据，30% 数据时稳定超越 Task Arithmetic，显著减少数据标注成本。

**计算成本**：单次迭代耗时约 70 秒（T5-base）/170 秒（T5-large），远低于重新训练多任务模型的计算量。

这些优势使其成为实际场景中高效合并现有模型的理想选择。



# 《模型合并：从基础到动态 Fisher 加权合并》

论文的第 2 章 “Model Merging Revisit” 和第 3 章 “Dynamic Fisher-weighted Merging” 主要围绕模型合并展开，涵盖从基础概念、已有方法到创新的动态 Fisher 加权合并（DF-Merge）方法，为构建高效多任务模型提供了理论与实践依据。

**第 2 章：模型合并再探讨**

**符号表示**：用$net(\theta)$代表由参数$\theta\in\mathbb{R}^{d}$构建的神经网络。在模型合并场景中，存在$T$个任务特定模型$\{net(\theta_{i})\}_{i = 1}^{T}$，它们都从同一个预训练模型$net(\theta_{pre})$初始化，各自在对应的任务数据集$D_{i}=\{x_{i}^{(j)}, y_{i}^{(j)}\}_{j = 1}^{N_{i}}$上微调 ，$N_{i}$是$D_{i}$的样本数量。模型合并的核心目标就是打造一个能胜任所有任务的多任务模型$net(\theta^{*})$。

**任务算术（TA）**：Ilharco 等人提出任务向量概念，第$i$个任务的任务向量$\tau_{i}=\theta_{i}-\theta_{pre}$，它指示了在参数空间里能提升预训练模型在该任务性能的方向。基于此，通过对任务向量进行算术运算，能引导预训练模型在不同任务上的表现。在模型级的模型合并中，有公式$\theta_{new }=\theta_{pre }+\lambda \sum_{i = 1}^{M} \Phi(\tau_{i})$，其中$\lambda$是缩放系数，用来调整任务向量对预训练模型参数的影响程度，$\Phi$代表像修剪、选择或随机失活等对任务向量的额外操作，目的是减少微调模型间的参数干扰。进一步推广，得到通用任务算术（GTA），公式为$\theta_{new }=\theta_{pre }+\sum_{i = 1}^{M} \lambda_{i} \Phi(\tau_{i})$，通过设置多个系数$\lambda_{i}$，可以更灵活地控制不同任务向量的作用。

**Fisher 信息**：损失函数$\ell(\theta)$在$\theta$处的局部曲率由其二阶导数$\nabla^{2} \ell(\theta)$（即 Hessian 矩阵$H_{\theta}\in\mathbb{R}^{d×d}$）体现。在数据分布$p_{\theta}(x, y)$上对$H_{\theta}$取期望，能反映出损失函数对参数变化的敏感程度，敏感程度高意味着该参数更重要。假设模型使用负对数似然损失$\ell(\theta)=-\log p(y | x, \theta)$，则 Fisher 信息（FI）表示为$F_{\theta}=\underset{x \sim q(x)}{\mathbb{E}}\left[\underset{y \sim p_{\theta}(y | x)}{\mathbb{E}} \nabla_{\theta} \ell(\theta) \nabla_{\theta} \ell(\theta)^{\top}\right]$ 。由于计算$F_{\theta}$中对输入分布$x\sim q(x)$的期望较困难，实际常采用经验 Fisher 信息$\hat{F}_{\theta}=\frac{1}{N} \sum_{i = 1}^{N}\left[\underset{y \sim p_{\theta}\left(y | x^{(i)}\right)}{\mathbb{E}} \nabla_{\theta} \ell(\theta) \nabla_{\theta} \ell(\theta)^{\top}\right]$来近似，其中对$y$的期望是基于模型的预测分布$y\sim p_{\theta}(y | x)$ ，而非真实标签。

**从几何角度看 Fisher 合并**：Tam 等人对 Fisher 合并进行几何分析，其公式为$\theta^{*}=\left(\sum_{i = 1}^{M} Q_{i} \Lambda_{i} Q_{i}^{\top}\right)^{-1}\left(\sum_{i = 1}^{M} Q_{i} \Lambda_{i} Q_{i}^{\top} \theta_{i}\right)$，$Q_{i} \Lambda_{i} Q_{i}^{\top}$是$F_{\theta_{i}}$的特征分解。它会对$\theta_{i}$中 “重要” 的特征向量分量加重权重，这样在合并时就能保留有用参数。从几何目标$g(\theta)=\underset{\theta}{arg min } \sum_{i = 1}^{M}\left\| \Lambda_{i}^{1 / 2}\left(Q_{i}^{\top} \theta_{i}-Q_{i}^{\top} \theta\right)\right\| ^{2}$出发，该目标限制$\theta$沿着参数空间中损失不敏感的主方向移动，与较小特征值相关的特征向量就指示了这些方向。因为每个微调模型$\theta_{i}$是各自任务的局部最小值，沿着损失不敏感方向移动有助于避免$\theta$增加各任务的损失，从而平衡微调模型，找到所有任务共享的低损失区域。对$g(\theta)$求梯度并令其为 0，经过推导可得到与 Fisher 合并等价的结果。实际应用中，为降低计算复杂度，常用$F_{\theta_{i}}$的对角近似，这相当于假设参数之间相互独立（即$Q_{i}=I$ ）。

**第 3 章：动态 Fisher 加权合并**

**统一模型合并视角**：作者提出一个通用的模型合并函数$f(\lambda_{1}, ..., \lambda_{M} ; \theta_{1}, ..., \theta_{M})=\left(\sum_{i}^{M} C_{\theta_{i}}\right)^{-1}\left(M \sum_{i}^{M} C_{\theta_{i}} \cdot \lambda_{i} \tau_{i}\right)+\theta_{pre }$，这个函数将不同的模型合并方法统一起来。当$\lambda_{i}=1 / M$且$C_{\theta_{i}}=I$时，该函数就变成了简单的平均合并；当$C_{\theta_{i}}=I$时，对应通用任务算术（GTA）；当$C_{\theta_{i}}=diag(\hat{F}_{\theta_{i}})$且$\lambda_{i}=1 / M$时，则是 Fisher 合并。在此基础上，作者提出 DF - Merge 的合并函数$f=\left(\sum_{i}^{M} diag\left(\hat{F}_{\theta_{i}\left(\lambda_{i}\right)}\right)\right)^{-1}\left(M \sum_{i}^{M} diag\left(\hat{F}_{\theta_{i}\left(\lambda_{i}\right)}\right) \lambda_{i} \tau_{i}\right)+\theta_{pre }$ ，这里的$diag(\hat{F}_{\theta_{i}(\lambda_{i})})$是在$\theta_{i}(\lambda_{i}):=\lambda_{i} \tau_{i}+\theta_{pre }$处估计的对角 Fisher 信息。与固定$diag(\hat{F}_{\theta_{i}(1)})$的 Fisher 合并不同，DF - Merge 可以沿着连接$\theta_{pre }$和$\theta_{i}$的路径，根据不同的$\lambda_{i}$来估计 Fisher 信息，从而更灵活地融合模型。

**系数优化**：为确定能使验证集平均准确率最大化的系数$\{\lambda_{i}\}_{i = 1}^{M}$，作者采用贝叶斯优化方法。具体来说，利用高斯过程对返回标量指标（平均准确率）的黑盒函数$f_{b}(\lambda)$建模 ，这里$\lambda:=[\lambda_{1}, ..., \lambda_{M}]\in\mathbb{R}^{M}$。首先对$t$个初始随机观测点设置高斯过程先验$f_{b}\left(\lambda^{1: t}\right) \sim \mathcal{N}\left(\mu_{0}\left(\lambda^{1: t}\right), \sum_{0}\left(\lambda^{1: t}, \lambda^{1: t}\right)\right)$ ，其中$\lambda^{1: t}$是$t$个点$[\lambda^{1}, ..., \lambda^{t}]$的紧凑表示，$\mu_{0}$和$\sum_{0}$分别是均值函数和协方差函数。然后依据贝叶斯规则，更新下一点$f_{b}(\lambda^{t + 1})$的后验分布$f_{b}\left(\lambda^{t+1}\right) | f_{b}\left(\lambda^{1: t}\right) \sim \mathcal{N}\left(\mu_{t}\left(\lambda^{t+1}\right), \sigma_{t}^{2}\left(\lambda^{t+1}\right)\right)$ ，$\mu_{t}(\lambda^{t+1})$和$\sigma_{t}^{2}(\lambda^{t+1})$由特定公式计算。通过采集函数来确定下一个采样点$\lambda^{t + 1}$，在实验中考虑了预期改进（EI）和上置信界（UCB）两种采集函数。EI 会选择能使相对于当前最佳值$f_{b}^{*}(t)$的改进期望值最大化的$\lambda^{t+1}$ ，公式为$\underset{\lambda^{t+1}}{arg max } E_{f_{b}\left(\lambda^{t+1}\right)}\left[max \left(f_{b}\left(\lambda^{t+1}\right)-f_{b}^{*}(t), 0\right)\right]$；UCB 则选择使$\lambda^{t+1}$处置信区间峰值最大化的点 ，公式为$\underset{\lambda^{t+1}}{arg max } \mu_{t}\left(\lambda^{t+1}\right)+\beta^{1 / 2} \sigma_{t}\left(\lambda^{t+1}\right)$，其中$\beta$是用于平衡探索与利用的常数。不断重复采样过程，直到达到预先设定的迭代次数或者指标收敛。