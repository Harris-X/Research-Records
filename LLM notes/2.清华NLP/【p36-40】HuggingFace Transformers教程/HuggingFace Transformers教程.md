# HuggingFace Transformers教程

> https://helloai.blog.csdn.net/article/details/126474707
>
> https://colab.research.google.com/drive/1tcDiyHIKgEJp4TzGbGp27HYbdFWGolU_?usp=sharing#scrollTo

每一次都需要从头实现模型，对于使用预训练模型的人来说是一个非常繁重的工作。

我们就会希望有一个第三方的库能帮助我们：

- 轻松地复现各种预训练语言模型论文的结果
- 快速部署模型
- 自由地自定义模型

而HuggingFace就提供了这样一个库，叫`transformers`。
安装也非常简单`pip install transformers`。

### 使用Pipeline

这个库有一个非常重要的工具叫pipeline，它主要的使用场景是你希望使用现成的预训练好的模型来完成你的任务。

比如你希望有一个微调好的预训练模型来完成情感分析任务，那么只要输入任务名(sentiment-analysis)到`pipeline`中，那么它会根据任务名为你提供一个微调好的模型。

<img src="HuggingFace Transformers教程.assets/a7c26c4c2dff42108c52a9ca23ba5193.png" alt="在这里插入图片描述" style="zoom:50%;" />

再比如一个问答任务，只要输入问题和上下文。

<img src="HuggingFace Transformers教程.assets/e21909f6981b407cac550bb8074608f1.png" alt="在这里插入图片描述" style="zoom:50%;" />

### Tokenization

如果不想直接用微调好的模型，而是想在自己的数据上去微调，那么第一个要做的事情就是分词。
我们知道，不同的模型可能包含不同的分词技术。

<img src="HuggingFace Transformers教程.assets/04f503e774af4eca8bb1a0176abbc646.png" alt="在这里插入图片描述" style="zoom:50%;" />

但有了`transformers`之后，我们不需要担心这些具体的不同。只需要引入`AutoTokenizer`，它会帮你根据不同的模型自动选择对应的分词器。然后直接传入要分词的文本即可。

<img src="HuggingFace Transformers教程.assets/add80360e733408e838343345c6946d9.png" alt="在这里插入图片描述" style="zoom:50%;" />

### 常用API介绍

除了分词之外，要微调自己的数据或模型，你还需要了解一些API。比如加载一个预训练的模型。

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# 加载预训练的模型
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

```

然后对输入进行分词：

```python
inputs = tokenizer("Hello World!", return_tensors='pt')
```

接着将输入喂给模型：

```python
outputs = model(**inputs)
```

最后可能需要保存微调好的模型：

```python
model.save_pretrained('path_to_save_model')
```

实际上在训练时也有封装好的API：

```python
trainer = Trainer(
    model, # 模型
    args,  # 优化相关的参数
    train_dataset=encoded_dataset["train"],  # 训练数据集
    eval_dataset=encoded_dataset["validation"],  # 验证集
    tokenizer=tokenizer,  # 分词器
    compute_metrics=compute_metrics # 评估指标
)
trainer.train() # 开始训练
trainer.evaluate() # 开始评估

```

## 实战

Pipeline虽然好用，但是不支持微调。如果想加载预训练模型，并自己微调的话，我们需要额外写一些加载模型、数据处理的代码。

以BERT在GLUE的SST-2数据集进行情感分析为例，展示如何使用微调。

首选我们需要加载要微调的数据集。

### 加载数据集

HuggingFace还提供了`datasets`库，包含了主流的数据集，通过一行命令(`load_dataset`)就可以完成数据集的下载与加载，且能够加载该数据集对应的指标(metric)以便计算(`load_metric`)。在这个例子中，我们需要加载GLUE中的SST-2任务。

我们使用Google的Colab来完成本次实战。

```python
!pip install transformers datasets
```

首先我们要安装需要用到的这两个库。

然后加载GLUE中的SST-2任务。

```python
from datasets import load_dataset, load_metric
dataset = load_dataset("glue", "sst2")
metric = load_metric("glue", "sst2")
```

我们看一下这个`dataset`，可以看到数据集分为train, validation, test，其中每个集合中包含三个key，分别对应文本、标签以及编号

<img src="HuggingFace Transformers教程.assets/image-20231023114106588.png" alt="image-20231023114106588" style="zoom: 67%;" />

我们也可以看第一条训练数据长啥样：

<img src="HuggingFace Transformers教程.assets/image-20231023114206701.png" alt="image-20231023114206701" style="zoom:67%;" />

下面打印指标信息：

```py
metric
```

```
Metric(name: "glue", features: {'predictions': Value(dtype='int64', id=None), 'references': Value(dtype='int64', id=None)}, usage: """
Compute GLUE evaluation metric associated to each GLUE dataset.
Args:
    predictions: list of predictions to score.
        Each translation should be tokenized into a list of tokens.
    references: list of lists of references for each translation.
        Each reference should be tokenized into a list of tokens.
Returns: depending on the GLUE subset, one or several of:
    "accuracy": Accuracy
    "f1": F1 score
    "pearson": Pearson Correlation
    "spearmanr": Spearman Correlation
    "matthews_correlation": Matthew Correlation
Examples:

    >>> glue_metric = datasets.load_metric('glue', 'sst2')  # 'sst2' or any of ["mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "hans"]
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0}

    >>> glue_metric = datasets.load_metric('glue', 'mrpc')  # 'mrpc' or 'qqp'
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'accuracy': 1.0, 'f1': 1.0}

    >>> glue_metric = datasets.load_metric('glue', 'stsb')
    >>> references = [0., 1., 2., 3., 4., 5.]
    >>> predictions = [0., 1., 2., 3., 4., 5.]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print({"pearson": round(results["pearson"], 2), "spearmanr": round(results["spearmanr"], 2)})
    {'pearson': 1.0, 'spearmanr': 1.0}

    >>> glue_metric = datasets.load_metric('glue', 'cola')
    >>> references = [0, 1]
    >>> predictions = [0, 1]
    >>> results = glue_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'matthews_correlation': 1.0}
""", stored examples: 0)

```

可以看到SST-2的指标为准确率。

在我们有了模型的预测结果以及正确结果之后，我们可以通过调用`metric.compute`来方便地计算模型的表现。我们先随机生成一些数据来展示使用方法。

```py
import numpy as np

fake_preds = np.random.randint(0, 2, size=(64,))                  # 随机生成一些预测结果
fake_labels = np.random.randint(0, 2, size=(64,))                 # 随机生成一些标签
metric.compute(predictions=fake_preds, references=fake_labels)    # 将二者输入metric.compute中

```

```
{'accuracy': 0.609375}
```

这样，我们已经完成了数据集的下载、加载以及对应指标的准备。

### 分词

预训练模型并不直接接受文本作为输入，每个预训练模型都有自己的分词方式以及自己的词表，我们在使用某一模型时，需要：

1. 使用该模型的分词方式对数据进行分词
2. 使用该模型的词表，将分词之后的每个token转化成对应的id

除了token的id之外，预训练模型还需要其他的一些输入。例如BERT还需要`token_type_ids`、`attention_mask`等。

但这种繁琐的工作HuggingFace也为我们进行了简化，我们只需要加载想用的模型的分词器即可。

```py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

```

下面就可以将文本直接传给分词器实例`tokenzier`就能得到模型的输入，例如：

```py
tokenizer("Tsinghua University is located in Beijing.")
```

```
{'input_ids': [101, 24529, 2075, 14691, 2118, 2003, 2284, 1999, 7211, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

```

下面我们就开始利用分词器来定义处理数据函数，由于BERT只能处理长度不超过512的序列，因此我们指定`truncation=True`来截断过长的序列。

```py
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, max_length=512)

```

我们可以使用数据集中的前5条数据来检验一下处理结果。

```py
preprocess_function(dataset['train'][:5])

```

```
{'input_ids': [[101, 5342, 2047, 3595, 8496, 2013, 1996, 18643, 3197, 102], [101, 3397, 2053, 15966, 1010, 2069, 4450, 2098, 18201, 2015, 102], [101, 2008, 7459, 2049, 3494, 1998, 10639, 2015, 2242, 2738, 3376, 2055, 2529, 3267, 102], [101, 3464, 12580, 8510, 2000, 3961, 1996, 2168, 2802, 102], [101, 2006, 1996, 5409, 7195, 1011, 1997, 1011, 1996, 1011, 11265, 17811, 18856, 17322, 2015, 1996, 16587, 2071, 2852, 24225, 2039, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

可以看到处理结果为一个字典，包含`input_ids`，`token_type_ids`以及`attention_mask`。

那么我们现在就可以使用`preprocess_function`来处理整个数据集，这一过程可以借助`dataset.map`函数来实现，该函数能将我们自定义的处理函数用到数据集的所有数据上。此外，通过指定`batched=True`，可以实现多线程并行处理来加速。

```py
encoded_dataset = dataset.map(preprocess_function, batched=True)
```

查看一下`encoded_dataset`，我们可以发现`encoded_dataset`在原先的`dataset`基础上，多出了三个feature，分别就是`tokenizer`输出的三个结果：

```
DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 67349
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 872
    })
    test: Dataset({
        features: ['sentence', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1821
    })
})
```

再次查看第一条训练数据：

```py
encoded_dataset['train'][0]
```

```
{'sentence': 'hide new secretions from the parental units ',
 'label': 0,
 'idx': 0,
 'input_ids': [101, 5342, 2047, 3595, 8496, 2013, 1996, 18643, 3197, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

至此，我们将所有数据转化成了模型能接受的输入格式`(input_ids, token_type_ids, attention_mask)`。

### 微调模型

数据集已经准确完毕，我们可以开始微调模型了。
首先，我们需要利用`transformers`把预训练下载下来，同时由于SST-2的标签种类只有两种，因此我们指定`num_labels=2`。

```py
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

代码会输出一些像是报错的信息，不用担心。这是因为我们为了利用BERT来进行情感分类，舍弃了原先BERT用来做masked language modeling和句子关系预测的参数，替换为了一个新的分类层来进行训练。

下面，我们使用`Trainer`类来进行模型的微调。这里，我们设置它的各种参数如下：

```py
from transformers import TrainingArguments

batch_size=16
args = TrainingArguments(
    "bert-base-uncased-finetuned-sst2", # 训练的名称
    evaluation_strategy="epoch", # 在每个epoch结束的时候在validation集上测试模型效果
    save_strategy="epoch", # 在每个epoch结束的时候保存一个checkpoint
    learning_rate=2e-5, # 优化的学习率
    per_device_train_batch_size=batch_size, # 训练时每个gpu上的batch_size
    per_device_eval_batch_size=batch_size,  # 测试时每个gpu上的batch_size
    num_train_epochs=5, # 训练5个epoch
    weight_decay=0.01, # 优化时采用的weight_decay
    load_best_model_at_end=True, # 在训练结束后，加载训练过程中最好的参数
    metric_for_best_model="accuracy" # 以准确率作为指标
)

```

下面我们定义一个函数，告诉`Trainer`怎么计算指标：

```py
def compute_metrics(eval_pred):
    logits, labels = eval_pred                 # predictions: [batch_size,num_labels], labels:[batch_size,]
    predictions = np.argmax(logits, axis=1)    # 将概率最大的类别作为预测结果
    return metric.compute(predictions=predictions, references=labels)
```

现在我们可以定义出该`Trainer`类了：

```py
from transformers import Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

```

这里我们使用默认的选项，优化器是AdamW，scheduler是linear warmup。

接着，调用`train`方法就可以开始训练了。

```py
trainer.train()
```

```
The following columns in the training set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence, idx. If sentence, idx are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
/usr/local/lib/python3.7/dist-packages/transformers/optimization.py:310: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  FutureWarning,
***** Running training *****
  Num examples = 67349
  Num Epochs = 5
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 1
  Total optimization steps = 21050
 [21050/21050 48:49, Epoch 5/5]
Epoch	Training Loss	Validation Loss	Accuracy
1	0.177300	0.327943	0.916284
2	0.121800	0.339612	0.917431
3	0.089700	0.341416	0.918578
4	0.053900	0.441544	0.915138
5	0.030300	0.464400	0.910550
The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence, idx. If sentence, idx are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 872
  Batch size = 16
Saving model checkpoint to bert-base-uncased-finetuned-sst2/checkpoint-4210
Configuration saved in bert-base-uncased-finetuned-sst2/checkpoint-4210/config.json
Model weights saved in bert-base-uncased-finetuned-sst2/checkpoint-4210/pytorch_model.bin
tokenizer config file saved in bert-base-uncased-finetuned-sst2/checkpoint-4210/tokenizer_config.json
Special tokens file saved in bert-base-uncased-finetuned-sst2/checkpoint-4210/special_tokens_map.json
The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence, idx. If sentence, idx are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 872
  Batch size = 16
Saving model checkpoint to bert-base-uncased-finetuned-sst2/checkpoint-8420
Configuration saved in bert-base-uncased-finetuned-sst2/checkpoint-8420/config.json
Model weights saved in bert-base-uncased-finetuned-sst2/checkpoint-8420/pytorch_model.bin
tokenizer config file saved in bert-base-uncased-finetuned-sst2/checkpoint-8420/tokenizer_config.json
Special tokens file saved in bert-base-uncased-finetuned-sst2/checkpoint-8420/special_tokens_map.json
The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence, idx. If sentence, idx are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 872
  Batch size = 16
Saving model checkpoint to bert-base-uncased-finetuned-sst2/checkpoint-12630
Configuration saved in bert-base-uncased-finetuned-sst2/checkpoint-12630/config.json
Model weights saved in bert-base-uncased-finetuned-sst2/checkpoint-12630/pytorch_model.bin
tokenizer config file saved in bert-base-uncased-finetuned-sst2/checkpoint-12630/tokenizer_config.json
Special tokens file saved in bert-base-uncased-finetuned-sst2/checkpoint-12630/special_tokens_map.json
The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence, idx. If sentence, idx are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 872
  Batch size = 16
Saving model checkpoint to bert-base-uncased-finetuned-sst2/checkpoint-16840
Configuration saved in bert-base-uncased-finetuned-sst2/checkpoint-16840/config.json
Model weights saved in bert-base-uncased-finetuned-sst2/checkpoint-16840/pytorch_model.bin
tokenizer config file saved in bert-base-uncased-finetuned-sst2/checkpoint-16840/tokenizer_config.json
Special tokens file saved in bert-base-uncased-finetuned-sst2/checkpoint-16840/special_tokens_map.json
The following columns in the evaluation set don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: sentence, idx. If sentence, idx are not expected by `BertForSequenceClassification.forward`,  you can safely ignore this message.
***** Running Evaluation *****
  Num examples = 872
  Batch size = 16
Saving model checkpoint to bert-base-uncased-finetuned-sst2/checkpoint-21050
Configuration saved in bert-base-uncased-finetuned-sst2/checkpoint-21050/config.json
Model weights saved in bert-base-uncased-finetuned-sst2/checkpoint-21050/pytorch_model.bin
tokenizer config file saved in bert-base-uncased-finetuned-sst2/checkpoint-21050/tokenizer_config.json
Special tokens file saved in bert-base-uncased-finetuned-sst2/checkpoint-21050/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from bert-base-uncased-finetuned-sst2/checkpoint-12630 (score: 0.9185779816513762).
TrainOutput(global_step=21050, training_loss=0.10431196976727375, metrics={'train_runtime': 2929.8649, 'train_samples_per_second': 114.935, 'train_steps_per_second': 7.185, 'total_flos': 6090242903971080.0, 'train_loss': 0.10431196976727375, 'epoch': 5.0})
```

