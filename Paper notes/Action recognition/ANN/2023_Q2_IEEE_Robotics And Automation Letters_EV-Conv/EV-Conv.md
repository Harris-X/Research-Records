# Ev-Conv: Fast CNN Inference on Event Camera Inputs for High-Speed Robot Perception

> 我们观察到，来自事件相机的连续输入到CNN之间只有很小的差异。因此，我们建议对连续输入张量之间的差异，或增量进行推理。这使得所需浮点运算数量显著减少（因此推理延迟也减少），因为增量非常稀疏。我们设计Ev-Conv以利用事件相机增量的不规则稀疏性，并在所有网络层中保持这些增量的稀疏性。