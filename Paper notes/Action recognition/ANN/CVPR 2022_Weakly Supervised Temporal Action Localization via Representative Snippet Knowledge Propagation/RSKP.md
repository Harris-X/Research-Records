#### Weakly Supervised Temporal Action Localization via Representative Snippet Knowledge Propagation

- https://blog.csdn.net/weixin_44609958/article/details/140444781

Abstract
弱监督时间动作定位的目标是通过仅有的视频级类别标签，来定位动作的时间边界并同时识别其类别。许多现有方法试图生成伪标签以弥合分类和定位之间的差距，但通常只利用有限的上下文信息来生成伪标签。为了解决这个问题，我们提出了一种代表性片段总结与传播框架。**我们的方法旨在挖掘每个视频中的代表性片段，通过在视频片段之间传播信息来生成更好的伪标签**。对于每个视频，利用其自身的代表性片段以及来自记忆库的代表性片段，以视频内和视频间的方式传播，更新输入特征。伪标签由更新特征的时间类激活图生成，以纠正主分支的预测。我们的方法在两个基准数据集THUMOS14和ActivityNet1.3上表现优越，在THUMOS14上平均mAP指标提升了高达1.2%。