---
title: "Chapter8 RAG系统实战"
categories: 技术
tags: ["RAG", "实战项目", "系统构建", "智能问答", "菜谱推荐"]
id: "rag-practical-project"
date: 2025-12-25 18:18:18
recommend: false
top: true
---

## 第八章学习笔记：RAG系统实战

## 学习目标

1. 了解RAG系统实战项目的背景和目标
2. 掌握RAG系统的环境配置和项目架构设计
3. 学习数据准备模块的实现，包括父子文本块策略
4. 掌握索引构建与检索优化的方法
5. 理解生成集成与系统整合的实现
6. 了解RAG系统的优化方向

## 学习内容

### 一、环境配置与项目架构

#### 1.1 项目背景

本章实战项目是一个基于HowToCook开源菜谱项目的智能问答系统——"尝尝咸淡RAG系统"。该项目旨在解决"今天吃什么"的选择困难症，通过AI助手根据用户需求推荐菜品并提供制作方法。

HowToCook项目特点：
- 包含约300多个Markdown格式的菜谱文件
- 结构高度规整，每个文件都严格按照统一的格式组织内容
- 内容篇幅较短，单个菜谱通常在700字左右

#### 1.2 环境配置

**创建虚拟环境**：
```bash
# 使用conda创建环境
conda create -n cook-rag-1 python=3.12.7
conda activate cook-rag-1
```

**安装核心依赖**：
```bash
cd code/C8
pip install -r requirements.txt
```

**API配置**：
- 申请Kimi API Key：[Kimi API官网](https://platform.moonshot.cn/console/api-keys)
- 配置MOONSHOT_API_KEY环境变量

#### 1.3 项目架构设计

**项目目标**：
- 询问具体菜品的制作方法："宫保鸡丁怎么做？"
- 寻求菜品推荐："推荐几个简单的素菜"
- 获取食材信息："红烧肉需要什么食材？"

**数据分析**：
- 文档结构高度规整，包含"必备原料和工具"、"计算"、"操作"、"附加内容"等部分
- 适合采用Markdown结构分块策略
- 需要解决上下文信息不完整的问题

**父子文本块策略**：
- **检索阶段**：使用小的子块进行精确匹配，提高检索准确性
- **生成阶段**：传递完整的父文档给LLM，确保上下文完整性
- **智能去重**：当检索到同一道菜的多个子块时，合并为一个完整菜谱

**整体架构**：
系统采用模块化设计，包含四个主要流程：
1. 数据准备模块
2. 索引构建模块
3. 检索优化模块
4. 生成集成模块

**项目结构**：
```
code/C8/
├── config.py                   # 配置管理
├── main.py                     # 主程序入口
├── requirements.txt            # 依赖列表
├── rag_modules/               # 核心模块
│   ├── __init__.py
│   ├── data_preparation.py    # 数据准备模块
│   ├── index_construction.py  # 索引构建模块
│   ├── retrieval_optimization.py # 检索优化模块
│   └── generation_integration.py # 生成集成模块
└── vector_index/              # 向量索引缓存（自动生成）
```

### 二、数据准备模块实现

#### 2.1 核心设计

**父子文本块映射关系**：
```
父文档（完整菜谱）
├── 子块1：菜品介绍 + 难度评级
├── 子块2：必备原料和工具
├── 子块3：计算（用量配比）
├── 子块4：操作（制作步骤）
└── 子块5：附加内容（变化做法）
```

**基本流程**：
- **检索阶段**：使用小的子块进行精确匹配，提高检索准确性
- **生成阶段**：传递完整的父文档给LLM，确保上下文完整性
- **智能去重**：当检索到同一道菜的多个子块时，合并为一个完整菜谱

**元数据增强**：
- **菜品分类**：从文件路径推断（荤菜、素菜、汤品等）
- **难度等级**：从内容中的星级标记提取
- **菜品名称**：从文件名提取
- **文档关系**：建立父子文档的ID映射关系

#### 2.2 模块实现详解

**类结构设计**：
```python
class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射
```

**文档加载实现**：
- 使用`rglob("*.md")`递归查找所有Markdown文件
- 为每个父文档分配唯一ID，建立父子关系
- 标记文档类型为"parent"，便于区分父子文档

**元数据增强**：
- **分类推断**: 从HowToCook项目的目录结构推断菜品分类
- **难度提取**: 从内容中的星级标记自动提取难度等级
- **名称提取**: 直接使用文件名作为菜品名称

**Markdown结构分块**：
- 使用MarkdownHeaderTextSplitter按照标题结构进行分块
- 定义三级标题分割：`#`、`##`、`###`
- 保留标题信息，设置`strip_headers=False`
- 为每个子块建立与父文档的关系

**智能去重**：
- 统计每个父文档被匹配的次数（相关性指标）
- 按相关性排序：匹配子块越多的菜谱排名越靠前
- 去重输出：每个菜谱只输出一次完整文档

### 三、索引构建与检索优化

#### 3.1 核心设计

**索引构建**：
- 选择BGE-small-zh-v1.5作为嵌入模型
- 使用FAISS作为向量数据库
- 实现索引缓存机制，提升系统启动速度

**混合检索**：
- **向量检索**：基于语义相似度，擅长理解查询意图
- **BM25检索**：基于关键词匹配，擅长精确匹配
- **RRF融合**：使用Reciprocal Rank Fusion算法综合两种检索结果

#### 3.2 索引构建模块

**类结构设计**：
```python
class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 index_save_path: str = "./vector_index"):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()
```

**嵌入模型初始化**：
- 使用HuggingFaceEmbeddings初始化BGE模型
- 设置`normalize_embeddings=True`进行向量归一化

**向量索引构建**：
- 提取文本内容和元数据
- 使用FAISS.from_texts构建向量索引
- 保存文本内容和元数据信息，支持高效检索

**索引缓存机制**：
- 首次构建后保存FAISS索引到本地
- 后续启动时直接加载已有索引
- 将启动时间从几分钟缩短到几秒钟

#### 3.3 检索优化模块

**类结构设计**：
```python
class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()
```

**检索器设置**：
- **向量检索器**：设置search_type="similarity"，search_kwargs={"k": 5}
- **BM25检索器**：使用BM25Retriever.from_documents创建，设置k=5

**RRF混合检索**：
- 分别获取向量检索和BM25检索结果
- 使用RRF算法计算综合分数：`score = 1 / (k + rank + 1)`，其中k=60
- 按RRF分数排序，返回top_k结果

**检索优势对比**：
- **向量检索的优势**：理解语义相似性、处理同义词和近义词、理解用户意图
- **BM25检索的优势**：精确匹配菜名、匹配具体食材、处理专业术语

**元数据过滤检索**：
- 支持按菜品分类、难度等级等条件进行筛选检索
- 扩大检索范围（top_k * 3）后再过滤，确保结果数量充足

### 四、生成集成与系统整合

#### 4.1 生成集成模块

**设计思路**：
- **智能查询路由**：根据用户查询自动判断是列表查询、详细查询还是一般查询
- **查询重写优化**：对模糊不清的查询进行智能重写，提升检索效果
- **多模式生成**：列表模式、详细模式、基础模式

**类结构设计**：
```python
class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "kimi-k2-0711-preview", 
                 temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
```

**查询路由实现**：
- 使用LLM自动判断查询意图，比简单的关键词匹配更准确
- 三种查询类型：
  - 'list'：用户想要获取菜品列表或推荐
  - 'detail'：用户想要具体的制作方法或详细信息
  - 'general'：其他一般性问题

**查询重写优化**：
- 使用LLM分析查询是否需要重写
- 具体明确的查询（如"宫保鸡丁怎么做"）保持原样
- 模糊查询（如"做菜"、"推荐个菜"）进行重写优化

**多模式生成**：
- **列表模式**：提取菜品名称，构建简洁的列表回答
- **详细模式**：使用结构化提示词，包含菜品介绍、所需食材、制作步骤、制作技巧
- **基础模式**：适用于一般性问题，提供常规回答

#### 4.2 系统整合

**主系统类设计**：
```python
class RecipeRAGSystem:
    """食谱RAG系统主类"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
```

**系统初始化流程**：
- 按依赖关系有序初始化各个模块
- 检查数据路径和API密钥是否存在

**知识库构建流程**：
- 尝试加载已保存的索引
- 如果索引不存在，构建新索引并保存
- 初始化检索优化模块

**智能问答流程**：
1. 查询路由：判断查询类型
2. 查询重写：根据路由类型决定是否重写
3. 混合检索：使用RRF算法获取相关子块
4. 父子文档处理：获取完整父文档并智能去重
5. 多模式生成：根据路由类型选择生成模式

**交互式问答**：
- 提供完整的命令行交互界面
- 支持流式输出和普通输出两种模式
- 实时显示生成过程，提升用户体验

#### 4.3 优化方向

**集成图数据库**：
- 将食谱数据构建为知识图谱
- 揭示食材、菜品与烹饪方法间的复杂关联
- 支持复杂关系查询和基于图的智能推荐

**融合多模态数据**：
- 结合菜品图片等视觉信息
- 利用多模态模型进行图文联合检索
- 支持视觉搜索和图像识别食材推荐

**增强专业知识**：
- 集成营养成分数据库
- 烹饪技巧知识图谱
- 食材替换规则库等外部知识源
- 提供精准的营养分析和专业烹饪指导

## 学习总结

本章通过一个完整的实战项目——"尝尝咸淡RAG系统"，展示了如何将前面学到的RAG知识应用到实际项目中。项目基于HowToCook菜谱数据，构建了一个智能问答系统，能够根据用户需求推荐菜品并提供制作方法。

项目采用了模块化设计，包含四个核心模块：数据准备模块、索引构建模块、检索优化模块和生成集成模块。关键技术点包括：

1. **父子文本块策略**：使用小块进行精确检索，大块保证生成质量
2. **索引缓存机制**：首次构建后保存索引，大幅提升系统启动速度
3. **RRF混合检索**：结合向量检索和BM25检索的优势，提高检索准确性
4. **智能查询路由**：根据用户意图自动选择最适合的处理策略
5. **多模式生成**：针对不同类型的查询提供不同格式的回答

通过这个项目，我们不仅学习了RAG系统的具体实现方法，还了解了如何解决实际开发中的各种挑战，如上下文完整性、检索精确性、系统性能优化等。这些经验对于构建其他类型的RAG系统也具有重要的参考价值。



> 参考教程： https://datawhalechina.github.io/all-in-rag/