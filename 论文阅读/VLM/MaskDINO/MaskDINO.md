# [Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation](https://www.cnblogs.com/lucifer1997/p/18211399)

**Abstract**：在本文中，我们提出了一个统一的目标检测和分割框架Mask DINO。Mask DINO通过添加一个支持所有图像分割任务（例如，全景和语义）的掩码预测分支来扩展DINO（具有改进的去噪锚框的DETR）。它利用DINO的查询嵌入来点积高分辨率的像素嵌入图来预测一组二值掩码。DINO中的一些关键组件通过共享的架构和训练过程进行扩展，用于分割。Mask DINO简单、高效、可扩展，可以受益于联合大规模检测和分割数据集。我们的实验表明，Mask DINO在ResNet-50主干和SwinL主干的预训练模型上都显著优于所有现有的专业分割方法。值得注意的是，Mask DINO在10亿个参数下的模型中建立了迄今为止实例分割（COCO上的54.5 AP）、全景分割（COCO上的59.4 PQ）和语义分割（ADE20K上的60.8 mIoU）的最佳结果。代码位于https://github.com/IDEA-Research/MaskDINO。

