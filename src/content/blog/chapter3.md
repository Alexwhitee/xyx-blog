
---
title: "Chapter3 向量嵌入与向量数据库"
categories: 技术
tags: ["RAG", "向量嵌入", "向量数据库", "多模态", "Milvus"]
id: "rag-embeddings-vector-db"
date: 2025-12-20 18:18:18
recommend: false
top: true
---

## 第三章学习笔记

## 第一节 向量嵌入

### 一、向量嵌入基础

#### 1.1 基础概念

**向量嵌入（Embedding）**是一种将真实世界中复杂、高维的数据对象（如文本、图像、音频、视频等）转换为数学上易于处理的、低维、稠密的连续数值向量的技术。

- **数据对象**：任何信息，如文本"你好世界"，或一张猫的图片
- **Embedding 模型**：一个深度学习模型，负责接收数据对象并进行转换
- **输出向量**：一个固定长度的一维数组，例如 `[0.16, 0.29, -0.88, ...]`

#### 1.1.2 向量空间的语义表示

Embedding 的真正意义在于，它产生的向量不是随机数值的堆砌，而是对数据**语义**的数学编码。

**核心原则**：在 Embedding 构建的向量空间中，语义上相似的对象，其对应的向量在空间中的距离会更近；而语义上不相关的对象，它们的向量距离会更远。

**关键度量**：
- **余弦相似度 (Cosine Similarity)**：计算两个向量夹角的余弦值。值越接近 1，代表方向越一致，语义越相似。这是最常用的度量方式。
- **点积 (Dot Product)**：计算两个向量的乘积和。在向量归一化后，点积等价于余弦相似度。
- **欧氏距离 (Euclidean Distance)**：计算两个向量在空间中的直线距离。距离越小，语义越相似。

### 1.2 Embedding 在 RAG 中的作用

#### 1.2.1 语义检索的基础

RAG 的"检索"环节通常以基于 Embedding 的语义搜索为核心。通用流程如下：

1. **离线索引构建**：将知识库内文档切分后，使用 Embedding 模型将每个文档块（Chunk）转换为向量，存入专门的向量数据库中。
2. **在线查询检索**：当用户提出问题时，使用**同一个** Embedding 模型将用户的问题也转换为一个向量。
3. **相似度计算**：在向量数据库中，计算"问题向量"与所有"文档块向量"的相似度。
4. **召回上下文**：选取相似度最高的 Top-K 个文档块，作为补充的上下文信息。

#### 1.2.2 决定检索质量的关键

Embedding 的质量直接决定了 RAG 检索召回内容的准确性与相关性。一个优秀的 Embedding 模型能够精准捕捉问题和文档之间的深层语义联系，即使用户的提问和原文的表述不完全一致。

### 二、Embedding 技术发展

#### 2.1 静态词嵌入：上下文无关的表示

- **代表模型**：Word2Vec (2013), GloVe (2014)
- **主要原理**：为词汇表中的每个单词生成一个固定的、与上下文无关的向量
- **局限性**：无法处理一词多义问题

#### 2.2 动态上下文嵌入

2017年，`Transformer` 架构的诞生带来了自注意力机制（Self-Attention），它允许模型在生成一个词的向量时，动态地考虑句子中所有其他词的影响。基于此，2018年 `BERT` 模型利用 `Transformer` 的编码器，通过掩码语言模型（MLM）等自监督任务进行预训练，生成了深度上下文相关的嵌入。

#### 2.3 RAG 对嵌入技术的新要求

- **领域自适应能力**：通用的嵌入模型在专业领域（如法律、医疗）往往表现不佳
- **多粒度与多模态支持**：RAG 系统需要处理不同长度和类型的输入数据
- **检索效率与混合检索**：嵌入向量的维度和模型大小直接影响存储成本和检索速度

### 三、嵌入模型训练原理

#### 3.1 主要训练任务

**任务一：掩码语言模型 (Masked Language Model, MLM)**
- 随机地将输入句子中 15% 的词元（Token）替换为一个特殊的 `[MASK]` 标记
- 让模型去预测这些被遮盖住的原始词元是什么
- 目标：通过这个任务，模型被迫学习每个词元与其上下文之间的关系

**任务二：下一句预测 (Next Sentence Prediction, NSP)**
- 构造训练样本，每个样本包含两个句子 A 和 B
- 其中 50% 的样本，B 是 A 的真实下一句（IsNext）；另外 50% 的样本，B 是从语料库中随机抽取的句子（NotNext）
- 让模型判断 B 是否是 A 的下一句
- 目标：这个任务让模型学习句子与句子之间的逻辑关系

#### 3.2 效果增强策略

- **度量学习 (Metric Learning)**：直接以"相似度"作为优化目标，让"正例对"的向量表示在空间中被"拉近"，而"负例对"的向量表示被"推远"

- **对比学习 (Contrastive Learning)**：在向量空间中，将相似的样本"拉近"，将不相似的样本"推远"。构建一个三元组（Anchor, Positive, Negative），训练的目标是让 `distance(Anchor, Positive)` 尽可能小，同时让 `distance(Anchor, Negative)` 尽可能大

### 四、嵌入模型选型指南

#### 4.1 从 MTEB 排行榜开始

**MTEB (Massive Text Embedding Benchmark)** 是一个由 Hugging Face 维护的、全面的文本嵌入模型评测基准。它涵盖了分类、聚类、检索、排序等多种任务，并提供了公开的排行榜。

关键评估维度：
- **横轴 - 模型参数量**：代表了模型的大小，参数量越大的模型潜在能力越强
- **纵轴 - 平均任务得分**：代表了模型的综合性能
- **气泡大小 - 嵌入维度**：代表了模型输出向量的维度
- **气泡颜色 - 最大处理长度**：代表了模型能处理的文本长度上限

#### 4.2 关键评估维度

1. **任务 (Task)**：对于 RAG 应用，需要重点关注模型在 `Retrieval` (检索) 任务下的排名
2. **语言 (Language)**：模型是否支持你的业务数据所使用的语言
3. **模型大小 (Size)**：模型越大，通常性能越好，但对硬件要求也越高
4. **维度 (Dimensions)**：向量维度越高，能编码的信息越丰富，但也会占用更多的存储空间
5. **最大 Token 数 (Max Tokens)**：这决定了模型能处理的文本长度上限
6. **得分与机构 (Score & Publisher)**：结合模型的得分排名和其发布机构的声誉
7. **成本 (Cost)**：API 服务的调用成本或自部署开源模型的资源消耗

#### 4.3 迭代测试与优化

1. **确定基线 (Baseline)**：根据上述维度，选择几个符合要求的模型作为初始基准模型
2. **构建私有评测集**：根据真实业务数据，手动创建一批高质量的评测样本
3. **迭代优化**：使用基线模型在私有评测集上运行，评估其召回的准确率和相关性，然后通过几轮的对比测试和迭代优化，选出最佳模型

## 第二节 多模态嵌入

### 一、为什么需要多模态嵌入？

现实世界的信息是多模态的，包含图像、音频、视频等。传统的文本嵌入无法理解"那张有红色汽车的图片"这样的查询，因为文本向量和图像向量处于相互隔离的空间，存在一堵"模态墙"。

**多模态嵌入 (Multimodal Embedding)** 的目标正是为了打破这堵墙。其目的是将不同类型的数据（如图像和文本）映射到**同一个共享的向量空间**。

### 二、CLIP 模型浅析

OpenAI 的 **CLIP (Contrastive Language-Image Pre-training)** 是一个很有影响力的模型，它为多模态嵌入定义了一个有效的范式。

CLIP 采用**双编码器架构 (Dual-Encoder Architecture)**，包含一个图像编码器和一个文本编码器，分别将图像和文本映射到同一个共享的向量空间中。

CLIP 在训练时采用了**对比学习 (Contrastive Learning)** 策略：最大化正确图文对的向量相似度，同时最小化所有错误配对的相似度。这种大规模的对比学习赋予了 CLIP 有效的**零样本（Zero-shot）识别能力**。

### 三、常用多模态嵌入模型

#### BGE-M3 模型

由北京智源人工智能研究院（BAAI）开发的 **BGE-M3** 是一个很有代表性的现代多模态嵌入模型。它的核心特性可以概括为"M3"：

- **多语言性 (Multi-Linguality)**：原生支持超过 100 种语言的文本与图像处理
- **多功能性 (Multi-Functionality)**：同时支持密集检索、多向量检索和稀疏检索
- **多粒度性 (Multi-Granularity)**：能够有效处理从短句到长达 8192 个 token 的长文档

在技术架构上，BGE-M3 采用了基于 XLM-RoBERTa 优化的联合编码器，并采用**网格嵌入 (Grid-Based Embeddings)**，将图像分割为多个网格单元并独立编码，提升了模型对图像局部细节的捕捉能力。

### 四、代码示例

#### 4.1 环境准备

1. **安装 visual_bge 模块**
   ```bash
   cd code/C3/visual_bge
   pip install -e .
   cd ..
   ```

2. **下载模型权重**
   ```bash
   python download_model.py
   ```

#### 4.2 基础示例

```python
import os
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
model.eval()

with torch.no_grad():
    text_emb = model.encode(text="datawhale开源组织的logo")
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")

# 计算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")
```

**代码解读**：
- `Visualized_BGE` 是通过将图像token嵌入集成到BGE文本嵌入框架中构建的通用多模态嵌入模型
- 支持纯文本编码、纯图像编码和图文联合编码
- 主要用于混合模态检索任务，包括多模态知识检索、组合图像检索等

## 第三节 向量数据库

### 一、向量数据库的作用

#### 1.1 向量数据库主要功能

1. **高效的相似性搜索**：利用专门的索引技术（如 HNSW, IVF），能够在数十亿级别的向量中实现毫秒级的近似最近邻（ANN）查询
2. **高维数据存储与管理**：专门为存储高维向量而优化，支持对向量数据进行增、删、改、查等基本操作
3. **丰富的查询能力**：支持按标量字段过滤查询、范围查询和聚类分析等
4. **可扩展与高可用**：采用分布式架构，具备良好的水平扩展能力和容错性
5. **数据与模型生态集成**：与主流的 AI 框架无缝集成

#### 1.2 向量数据库 vs 传统数据库

| **维度** | **向量数据库** | **传统数据库 (RDBMS)** |
| :--- | :--- | :--- |
| **核心数据类型** | 高维向量 (Embeddings) | 结构化数据 (文本、数字、日期) |
| **查询方式** | **相似性搜索** (ANN) | **精确匹配** |
| **索引机制** | HNSW, IVF, LSH 等 ANN 索引 | B-Tree, Hash Index |
| **主要应用场景** | AI 应用、RAG、推荐系统 | 业务系统 (ERP, CRM)、金融交易 |
| **数据规模** | 轻松应对千亿级向量 | 通常在千万到亿级行数据 |
| **性能特点** | 高维数据检索性能极高 | 结构化数据查询快，高维数据查询性能下降 |
| **一致性** | 通常为最终一致性 | 强一致性 (ACID 事务) |

向量数据库和传统数据库是**互补关系**，在构建现代 AI 应用时，通常会将两者结合使用。

### 二、工作原理

向量数据库通常采用四层架构：

1. **存储层**：存储向量数据和元数据，优化存储效率，支持分布式存储
2. **索引层**：维护索引算法（HNSW、LSH、PQ等），创建和优化索引
3. **查询层**：处理查询请求，支持混合查询，实现查询优化
4. **服务层**：管理客户端连接，提供监控和日志，实现安全管理

主要技术手段包括：
- **基于树的方法**：如 Annoy 使用的随机投影树
- **基于哈希的方法**：如 LSH（局部敏感哈希）
- **基于图的方法**：如 HNSW（分层可导航小世界图）
- **基于量化的方法**：如 Faiss 的 IVF 和 PQ

### 三、主流向量数据库介绍

**Pinecone**：完全托管的向量数据库服务，采用Serverless架构设计，提供存储计算分离、自动扩展和负载均衡等企业级特性

**Milvus**：开源的分布式向量数据库，采用分布式架构设计，支持GPU加速和多种索引算法，能够处理亿级向量检索

**Qdrant**：高性能的开源向量数据库，采用Rust开发，支持二进制量化技术，提供多种索引策略和向量混合搜索功能

**Weaviate**：支持GraphQL的AI集成向量数据库，提供20+AI模块和多模态支持，采用GraphQL API设计

**Chroma**：轻量级的开源向量数据库，采用本地优先设计，无依赖，提供零配置安装、本地运行和低资源消耗等特性

**选择建议**：
- **新手入门/小型项目**：从 `ChromaDB` 或 `FAISS` 开始
- **生产环境/大规模应用**：考虑 `Milvus`、`Weaviate` 或云服务 `Pinecone`

### 四、本地向量存储：以 FAISS 为例

FAISS (Facebook AI Similarity Search) 是一个由 Facebook AI Research 开发的高性能库，专门用于高效的相似性搜索和密集向量聚类。

#### 4.1 基础示例(FAISS)

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
docs = [Document(page_content=t) for t in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 创建向量存储并保存到本地
vectorstore = FAISS.from_documents(docs, embeddings)

local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)

# 3. 加载索引并执行查询
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

# 相似性搜索
query = "FAISS是做什么的？"
results = loaded_vectorstore.similarity_search(query, k=1)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")
```

#### 4.2 索引创建实现细节

通过深入 LangChain 源码，索引创建是一个分层、解耦的过程：

1. **`from_documents` (封装层)**：从输入的 `Document` 对象列表中提取出纯文本内容和元数据
2. **`from_texts` (向量化入口)**：调用 `embedding.embed_documents(texts)`，将所有文本批量转换为向量
3. **`__from` (构建索引框架)**：搭建 FAISS 向量存储的"空框架"，初始化空的 FAISS 索引结构
4. **`__add` (填充数据)**：执行数据添加操作，将向量列表转换为 FAISS 需要的 `numpy` 数组，并添加到 FAISS 索引中

## 第四节 Milvus介绍及多模态检索实践

### 一、简介

Milvus 是一个开源的、专为大规模向量相似性搜索和分析而设计的向量数据库。它采用云原生架构，具备高可用、高性能、易扩展的特性，能够处理十亿、百亿甚至更大规模的向量数据。

### 二、部署安装

#### 2.1 下载并启动 Milvus

**第一步：下载配置文件**
```bash
# macOS / Linux
wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Windows
Invoke-WebRequest -Uri "https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml" -OutFile "docker-compose.yml"
```

**第二步：启动 Milvus 服务**
```bash
docker compose up -d
```

#### 2.2 常用管理命令

- **停止服务**：`docker compose down`
- **彻底清理**：`docker compose down -v`

### 三、核心组件

#### 3.1 Collection (集合)

可以用一个图书馆的比喻来理解 Collection：

- **Collection (集合)**：相当于一个**图书馆**，是所有数据的顶层容器
- **Partition (分区)**：相当于图书馆里的**不同区域**，将数据物理隔离
- **Schema (模式)**：相当于图书馆的**图书卡片规则**，定义了数据必须登记哪些信息
- **Entity (实体)**：相当于**一本具体的书**，是数据本身
- **Alias (别名)**：相当于一个**动态的推荐书单**，可以指向某个具体的 Collection

##### 3.1.1 Schema

Schema 规定了 Collection 的数据结构，定义了其中包含的所有**字段 (Field)** 及其属性：

- **主键字段 (Primary Key Field)**：每个 Collection 必须有且仅有一个主键字段，用于唯一标识每一条数据
- **向量字段 (Vector Field)**：用于存储核心的向量数据，一个 Collection 可以有一个或多个向量字段
- **标量字段 (Scalar Field)**：用于存储除向量之外的元数据，如字符串、数字、布尔值、JSON 等

##### 3.1.2 Partition (分区)

**Partition** 是 Collection 内部的一个逻辑划分。使用分区的好处：

- **提升查询性能**：在查询时，可以指定只在一个或几个分区内进行搜索，大幅减少需要扫描的数据量
- **数据管理**：便于对部分数据进行批量操作

##### 3.1.3 Alias (别名)

**Alias** (别名) 是为 Collection 提供的一个"昵称"。使用别名的好处：

- **安全地更新数据**：创建新 Collection 并导入数据后，将别名从旧 Collection 原子性地切换到新 Collection 上
- **代码解耦**：整个切换过程对上层应用完全透明，无需修改任何代码或重启服务

#### 3.2 索引 (Index)

**索引 (Index)** 是加速检索的神经系统。对向量数据创建索引后，Milvus 可以极大地提升向量相似性搜索的速度。

##### 3.2.1 主要向量索引类型

- **FLAT (精确查找)**：暴力搜索，100% 的召回率，但速度慢，不适合海量数据
- **IVF 系列 (倒排文件索引)**：通过聚类将向量分成多个"桶"，先找相关桶再在桶内搜索，是性能和效果的平衡
- **HNSW (基于图的索引)**：构建多层邻近图，检索速度极快，召回率高，但内存占用大
- **DiskANN (基于磁盘的索引)**：为在 SSD 上运行而优化的图索引，支持超大规模数据集

##### 3.2.2 如何选择索引？

| 场景 | 推荐索引 | 备注 |
| :--- | :--- | :--- |
| 数据可完全载入内存，追求低延迟 | **HNSW** | 内存占用较大，但查询性能和召回率都很优秀 |
| 数据可完全载入内存，追求高吞吐 | **IVF_FLAT / IVF_SQ8** | 性能和资源消耗的平衡之选 |
| 数据量巨大，无法载入内存 | **DiskANN** | 在 SSD 上性能优异，专为海量数据设计 |
| 追求 100% 准确率，数据量不大 | **FLAT** | 暴力搜索，确保结果最精确 |

#### 3.3 检索

##### 3.3.1 基础向量检索 (ANN Search)

**近似最近邻 (Approximate Nearest Neighbor, ANN) 检索**利用预先构建好的索引，能够极速地从海量数据中找到与查询向量最相似的 Top-K 个结果。

主要参数：
- `anns_field`: 指定要在哪个向量字段上进行检索
- `data`: 传入一个或多个查询向量
- `limit` (或 `top_k`): 指定需要返回的最相似结果的数量
- `search_params`: 指定检索时使用的参数

##### 3.3.2 增强检索

**过滤检索 (Filtered Search)**
将**向量相似性检索**与**标量字段过滤**结合在一起，先根据过滤表达式筛选出符合条件的实体，然后仅在这个子集内执行 ANN 检索。

**范围检索 (Range Search)**
返回所有与查询向量的距离落在特定范围内的实体，用于人脸识别、异常检测等场景。

**多向量混合检索 (Hybrid Search)**
在一个请求中同时检索**多个向量字段**，并将结果智能地融合在一起。适用于多模态商品检索、增强型 RAG 等场景。

**分组检索 (Grouping Search)**
指定一个字段对结果进行分组，确保返回的结果中每个组只出现一次，且返回的是该组内与查询最相似的那个实体。

### 四、milvus多模态实践

#### 4.1 初始化与工具定义

```python
import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"

# 2. 定义工具 (编码器和可视化函数)
class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]
```

#### 4.2 创建 Collection

```python
# 3. 初始化客户端
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 4. 创建 Milvus Collection
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)

image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
dim = len(encoder.encode_image(image_list[0]))

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]

# 创建集合 Schema
schema = CollectionSchema(fields, description="多模态图文检索")
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
```

#### 4.3 准备并插入数据

```python
# 5. 准备并插入数据
data_to_insert = []
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")
```

#### 4.4 创建索引

```python
# 6. 创建索引
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
milvus_client.load_collection(collection_name=COLLECTION_NAME)
```

#### 4.5 执行多模态检索

```python
# 7. 执行多模态检索
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    output_fields=["image_path"],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}}
)[0]

retrieved_images = []
print("检索结果:")
for i, hit in enumerate(search_results):
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])
```

## 第五节 索引优化

### 一、上下文扩展

在RAG系统中，常常面临一个权衡问题：使用小块文本进行检索可以获得更高的精确度，但小块文本缺乏足够的上下文；而使用大块文本虽然上下文丰富，却容易引入噪音。

**句子窗口检索（Sentence Window Retrieval）**技术巧妙地结合了两种方法的优点：它在检索时聚焦于高度精确的单个句子，在送入LLM生成答案前，又智能地将上下文扩展回一个更宽的"窗口"。

#### 1.1 主要思路

句子窗口检索的思想可以概括为：**为检索精确性而索引小块，为上下文丰富性而检索大块**。

工作流程：
1. **索引阶段**：文档被分割成**单个句子**，每个句子都作为一个独立的"节点"存入向量数据库，同时存储其上下文窗口
2. **检索阶段**：在所有**单一句子节点**上执行相似度搜索，精确定位到与用户问题最相关的核心信息
3. **后处理阶段**：使用 `MetadataReplacementPostProcessor` 读取检索到的句子节点的元数据，用元数据中存储的**完整上下文窗口**来替换节点中原来的单一句子内容
4. **生成阶段**：将被替换了内容的、包含丰富上下文的节点传递给LLM

#### 1.2 代码实现

```python
# 1. 加载文档
documents = SimpleDirectoryReader(
    input_files=["../../data/C3/pdf/IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# 2. 创建节点与构建索引
# 2.1 句子窗口索引
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
sentence_nodes = node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(sentence_nodes)

# 2.2 常规分块索引 (基准)
base_parser = SentenceSplitter(chunk_size=512)
base_nodes = base_parser.get_nodes_from_documents(documents)
base_index = VectorStoreIndex(base_nodes)

# 3. 构建查询引擎
sentence_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)
base_query_engine = base_index.as_query_engine(similarity_top_k=2)

# 4. 执行查询并对比结果
query = "What are concerns surrounding AMOC?"
window_response = sentence_query_engine.query(query)
base_response = base_query_engine.query(query)
```

从结果对比可以看出，句子窗口检索的答案更详尽、更连贯，因为它通过"精确检索小文本块，再扩展上下文"的方式，为大语言模型提供了高度相关且信息丰富的上下文。

### 二、结构化索引

随着知识库的规模不断扩大，传统的RAG方法会遇到瓶颈。**结构化索引**通过在索引文本块的同时，为其附加结构化的**元数据（Metadata）**，可以实现"元数据过滤"和"向量搜索"的结合。

#### 2.1 代码实现：基于多表格的递归检索

在更复杂的场景中，结构化数据可能分布在多个来源中，需要一种更强大的策略：**递归检索**。它能实现"路由"功能，先将查询引导至正确的知识来源，然后再在该来源内部执行精确查询。

```python
# 1. 为每个工作表创建查询引擎和摘要节点
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

df_query_engines = {}
all_nodes = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    # 为当前工作表创建一个 PandasQueryEngine
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)
    # 为当前工作表创建一个摘要节点（IndexNode）
    year = sheet_name.replace('年份_', '')
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
    node = IndexNode(text=summary, index_id=sheet_name)
    all_nodes.append(node)
    # 存储工作表名称到其查询引擎的映射
    df_query_engines[sheet_name] = query_engine

# 2. 创建顶层索引（只包含摘要节点）
vector_index = VectorStoreIndex(all_nodes)

# 3. 创建递归检索器
vector_retriever = vector_index.as_retriever(similarity_top_k=1)
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_query_engines,
    verbose=True,
)

# 4. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# 5. 执行查询
query = "1994年评分人数最多的电影是哪一部？"
response = query_engine.query(query)
```

> ⚠️ **重要安全警告**：`PandasQueryEngine` 是一个实验性功能，具有潜在的安全风险。它的工作原理是让 LLM 生成 Python 代码，然后使用 `eval()` 函数在本地执行。**因此，强烈不建议在生产环境中使用此工具**。

#### 2.2 另一种实现方式

鉴于 `PandasQueryEngine` 的安全风险，可以采用一种更安全的方式来实现类似的多表格查询，思路是**将路由和检索彻底分离**：

1. **创建两个独立的向量索引**：
   - **摘要索引（用于路由）**：为每个Excel工作表创建一个简短的摘要性`Document`，构建轻量级的向量索引
   - **内容索引（用于问答）**：将每个工作表的实际数据转换为一个大的文本`Document`，并为其附加元数据标签

2. **执行两步查询**：
   - **第一步：路由**。在"摘要索引"中进行检索，确定目标工作表
   - **第二步：检索**。拿到目标后，在"内容索引"中进行检索，但附加**元数据过滤器**，强制要求只在指定工作表的文档中进行搜索

### 三、关于框架的思考

框架是加速开发的强大工具，但任何框架都有其设计边界和局限性。我们的目标不是成为一个熟练的"过桥者"，而是成为一个懂得如何设计和建造桥梁的"工程师"。

本教程选择的路径是：
1. **以原理为主**：优先关心"它是如何工作的？"而不是"我该调用哪个函数？"
2. **拥抱灵活性**：真实世界的业务需求往往比框架预设的场景更复杂
3. **培养解决问题的能力**：理解原理，则像是学会了烹饪的精髓，让你不仅能轻松地做出各种美食，还能创造新菜式

## 学习心得

通过第三章的学习，我对向量嵌入、多模态嵌入、向量数据库和索引优化有了全面的认识：

1. **向量嵌入是RAG系统的基石**：它将复杂的多模态数据转换为数学上易于处理的向量表示，实现了语义相似度的量化计算

2. **多模态嵌入打破了模态墙**：通过CLIP、BGE-M3等模型，实现了不同模态数据在同一向量空间的对齐，为跨模态检索奠定了基础

3. **向量数据库是高效检索的关键**：Milvus等专业向量数据库通过ANN索引、分布式架构等技术，实现了海量向量数据的高效存储和检索

4. **索引优化是提升RAG性能的重要手段**：句子窗口检索、结构化索引等技术通过优化检索策略和上下文组织，显著提升了检索质量和生成效果

5. **理解原理比掌握框架更重要**：虽然框架能加速开发，但只有理解了底层原理，才能在面对复杂需求时灵活应对，甚至创造新的解决方案

这些知识为构建高质量、高性能的RAG系统提供了坚实的技术基础。

> 参考教程： https://datawhalechina.github.io/all-in-rag/