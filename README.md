# 教学知识点与讲解内容一致性分析模型

## 1. 项目概述

本项目旨在提供一个完整、端到端的解决方案，用于训练一个深度学习模型，以分析**教学知识点**与**教学讲解内容**之间的一致性。其核心思想是通过先进的自然语言处理技术，将知识点和讲解内容都转换成高维度的数学向量（即“嵌入”或 Embeddings），然后通过计算这两个向量之间的相似度来判断它们内容的匹配程度。高相似度得分表示讲解内容与知识点要求一致，而低分则表示内容不匹配。

该工具可广泛应用于以下场景：
- **智能教育平台**：自动审核和校验课程内容是否符合其教学目标。
- **课程设计师**：确保新编写的教材或脚本紧密围绕核心知识点。
- **教师与学生**：快速验证学习资料或笔记是否与课程的知识点相关。

本项目基于强大的 `sentence-transformers` 开源库，并选择了 `paraphrase-multilingual-MiniLM-L12-v2` 作为基础模型进行微调（Fine-tuning）。该模型因其在中文及多语言语义相似度任务上的卓越性能而备受青睐。

## 2. 项目结构

为了便于理解和维护，项目采用了清晰、模块化的目录结构：

```
.
├── .gitignore               # 指定 Git 应忽略的文件和目录
├── data/
│   ├── train.csv            # 生成的训练数据集
│   └── test.csv             # 生成的测试数据集
├── src/
│   ├── generate_dataset.py  # 用于创建合成数据集的脚本
│   ├── dataset.py           # 用于加载数据的 PyTorch Dataset 类
│   ├── train.py             # 模型训练与评估的主脚本
│   └── inference.py         # 用于对新数据进行推理的脚本
├── requirements.txt         # 项目所有 Python 依赖库列表
└── README.md                # 本项目说明文档
```

## 3. 核心文件详解

### `src/generate_dataset.py`
此脚本用于创建本项目所需的中文合成数据集。
- **工作原理**: 脚本内部预定义了一个包含多个中文教学知识点（如“李白《静夜思》赏析”、“二次函数的定义与性质”等）及其相关讲解文本的字典。它通过以下方式生成数据对：
    - **正样本 (label=1)**: 将一个知识点与其相关的讲解文本配对。
    - **负样本 (label=0)**: 将一个知识点与一个**完全不相关**知识点的讲解文本进行配对。
- **输出**: 在 `data/` 目录下生成 `train.csv` 和 `test.csv` 两个文件。每个文件包含三列：`sentence1` (知识点), `sentence2` (讲解内容), 和 `label` (标签)，并以 `UTF-8` 编码保存。

### `src/dataset.py`
此脚本定义了一个名为 `SyllabusDataset` 的自定义 PyTorch `Dataset` 类，用于高效地从 CSV 文件加载数据。它将每一行数据封装成一个 `InputExample` 对象，供 `sentence-transformers` 库在训练时使用。

### `src/train.py`
这是项目的核心，负责模型的训练和评估。
- **模型加载**: 加载 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 模型作为微调的基础。
- **损失函数**: 采用 `CosineSimilarityLoss`，该损失函数旨在让模型预测的相似度得分尽可能接近真实的标签（正样本为1.0，负样本为0.0）。
- **集成评估**: 训练完成后，脚本会立即在测试集上评估模型性能，并打印出准确率、精确率、召回率和 F1-Score。
- **环境限制说明**: 由于执行环境的限制（无法保存大型模型文件），此脚本在训练后没有将模型保存到磁盘，而是直接在内存中进行评估。

### `src/inference.py`
此脚本演示了如何在实际应用中使用模型进行预测。
- **工作原理**: 它加载多语言模型，接收一个知识点和一个讲解文本作为输入，然后计算它们之间的余弦相似度。
- **输出**: 脚本会输出一个“一致性得分”。得分越接近1，表示两者内容越相关；得分越接近0，则表示内容差异越大。

## 4. 操作指南

### 步骤 1: 环境设置
建议使用虚拟环境。

```bash
# 安装所有依赖项
pip install -r requirements.txt
```

### 步骤 2: 生成中文数据集
运行以下命令来创建训练和测试数据。

```bash
python src/generate_dataset.py
```
执行成功后，`data/` 目录下会生成 `train.csv` 和 `test.csv`。

### 步骤 3: 训练与评估模型
运行主训练脚本来微调模型。

```bash
python -m src.train
```
终端将显示训练进度条，训练结束后会立即打印出评估结果。

### 步骤 4: 进行推理
使用推理脚本来测试模型对新文本对的分析能力。

```bash
python -m src.inference
```
脚本会使用预设的中文例子来展示模型的预测结果。

## 5. 自定义与扩展

### 使用您自己的数据
如果您想用自己的数据进行训练，只需按照相同的格式（三列：`sentence1`, `sentence2`, `label`）创建 `train.csv` 和 `test.csv` 文件，确保使用 `UTF-8` 编码，并替换掉 `data/` 目录下的旧文件即可。

### 保存并加载微调后的模型
若要在您自己的环境中保存模型，只需在 `src/train.py` 中为 `model.fit()` 方法添加 `output_path` 参数：

```python
# 文件: src/train.py
model.fit(..., output_path='my_chinese_model')
```
之后，在 `src/inference.py` 中加载您自己的微调模型：

```python
# 文件: src/inference.py
model = SentenceTransformer('my_chinese_model')
```
