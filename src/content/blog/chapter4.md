---
title: "Chapter4 高级检索技术"
categories: 技术
tags: ["RAG", "混合检索", "查询构建", "重排序", "检索优化"]
id: "rag-advanced-retrieval"
date: 2025-12-21 18:18:18
recommend: false
top: false
hide: false
---

## 第四章学习笔记

## 第一节：混合检索

### 一、稀疏向量 vs 密集向量

#### 1.1 稀疏向量

稀疏向量，也常被称为"词法向量"，是基于词频统计的传统信息检索方法的数学表示。它通常是一个维度极高（与词汇表大小相当）但绝大多数元素为零的向量。

**核心特点**：
- 采用精准的"词袋"匹配模型，将文档视为一堆词的集合
- 向量的每一个维度都直接对应一个具体的词，非零值代表该词在文档中的重要性
- 经典权重计算方法是 TF-IDF 和 BM25

**BM25 公式**：
$$ Score(Q, D) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})} $$

**优点**：
- 可解释性极强（每个维度都代表一个确切的词）
- 无需训练，能够实现关键词的精确匹配
- 对于专业术语和特定名词的检索效果好

**缺点**：
- 无法理解语义，例如无法识别"汽车"和"轿车"是同义词
- 存在"词汇鸿沟"问题

#### 1.2 密集向量

密集向量，也常被称为"语义向量"，是通过深度学习模型学习到的数据的低维、稠密的浮点数表示。

**核心特点**：
- 旨在将原始数据映射到一个连续的、充满意义的"语义空间"中捕捉"语义"或"概念"
- 在理想的语义空间中，向量之间的距离和方向代表了它们所表示概念之间的关系
- 代表包括 Word2Vec、GloVe、以及所有基于 Transformer 的模型生成的嵌入

**优点**：
- 能够理解同义词、近义词和上下文关系
- 泛化能力强，在语义搜索任务中表现卓越

**缺点**：
- 可解释性差（向量中的每个维度通常没有具体的物理意义）
- 需要大量数据和算力进行模型训练
- 对于未登录词（OOV）的处理相对困难

#### 1.3 实例对比

**稀疏向量表示**：
- 字典格式：`{"3": 5, "7": 9}`
- 坐标列表（COO）格式：`(8, [3, 7], [5, 9])`

**密集向量表示**：
- 数组格式：`[0.89, -0.12, 0.77, ..., -0.45]`
- 所有维度都有值，向量本身难以解读，但语义空间中的位置代表概念关系

### 二、混合检索

混合检索是一种结合了 **稀疏向量** 和 **密集向量** 优势的先进搜索技术，旨在同时利用稀疏向量的关键词精确匹配能力和密集向量的语义理解能力。

#### 2.1 技术原理与融合方法

##### 2.1.1 倒数排序融合 (Reciprocal Rank Fusion, RRF)

RRF 不关心不同检索系统的原始得分，只关心每个文档在各自结果集中的**排名**。

**计分公式**：
$$ RRF_{score}(d) = \sum_{i=1}^{k} \frac{1}{rank_i(d) + c} $$

其中：
- $d$ 是待评分的文档
- $k$ 是检索系统的数量（这里是2，即稀疏和密集）
- $rank_i(d)$ 是文档 $d$ 在第 $i$ 个检索系统中的排名
- $c$ 是一个常数（通常设为60）

##### 2.1.2 加权线性组合

$$ Hybrid_{score} = \alpha \cdot Dense_{score} + (1 - \alpha) \cdot Sparse_{score} $$

通过调整 `α` 的值，可以灵活地控制语义相似性与关键词匹配在最终排序中的贡献比例。

#### 2.2 优势与局限

| 优势 | 局限 |
| :--- | :--- |
| **召回率与准确率高**：能同时捕获关键词和语义，显著优于单一检索。 | **计算资源消耗大**：需要同时维护和查询两套索引。 |
| **灵活性强**：可通过融合策略和权重调整，适应不同业务场景。 | **参数调试复杂**：融合权重等超参数需要反复实验调优。 |
| **容错性好**：关键词检索可部分弥补向量模型对拼写错误或罕见词的敏感性。 | **可解释性仍是挑战**：融合后的结果排序理由难以直观分析。 |

### 三、代码实践：通过 Milvus 实现混合检索

#### 3.1 步骤一：定义 Collection

```python
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]
```

**fields字段类型分析**：
- **pk**: 主键设计，`auto_id=True` 让 Milvus 自动生成唯一标识
- **标量字段**: 7个VARCHAR字段用于存储元数据，`max_length` 根据实际数据分布优化存储
- **稀疏向量**: `SPARSE_FLOAT_VECTOR` 类型，存储关键词权重
- **密集向量**: `FLOAT_VECTOR` 类型，固定1024维，存储语义特征

#### 3.2 步骤二：BGE-M3 双向量生成

BGE-M3 能够同时生成稀疏向量和密集向量：

```python
embeddings = ef(docs)
sparse_vectors = embeddings["sparse"]    # 稀疏向量：词频统计
dense_vectors = embeddings["dense"]      # 密集向量：语义编码
```

#### 3.3 步骤三：实现混合检索

使用 RRF 算法进行混合检索：

```python
# 创建 RRF 融合器
rerank = RRFRanker(k=60)

# 创建搜索请求
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k)
sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=top_k)

# 执行混合搜索
results = collection.hybrid_search(
    [sparse_req, dense_req],
    rerank=rerank,
    limit=top_k,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]
```

**混合检索结果分析**：
- 密集向量搜索：侧重语义相似性，能理解"悬崖上的巨龙"的整体含义
- 稀疏向量搜索：侧重关键词匹配，更关注"悬崖"、"巨龙"等具体词汇
- 混合检索：结合两者优势，既考虑语义又考虑关键词匹配，结果更全面准确

## 第二节：查询构建

### 一、文本到元数据过滤器

**自查询检索器（Self-Query Retriever）** 是实现这一功能的核心组件。它的工作流程如下：

1. **定义元数据结构**：向LLM清晰地描述文档内容和每个元数据字段的含义及类型
2. **查询解析**：将查询分解为查询字符串和元数据过滤器
3. **执行查询**：将解析出的查询字符串和元数据过滤器发送给向量数据库

### 代码示例

```python
# 配置元数据字段信息
metadata_field_info = [
    AttributeInfo(name="title", description="视频标题（字符串）", type="string"),
    AttributeInfo(name="author", description="视频作者（字符串）", type="string"),
    AttributeInfo(name="view_count", description="视频观看次数（整数）", type="integer"),
    AttributeInfo(name="length", description="视频长度，以秒为单位的整数", type="integer")
]

# 创建自查询检索器
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)
```

**关键设计原则**：
1. **配置元数据字段**：通过 `AttributeInfo` 为每个元数据字段定义名称、类型和自然语言描述
2. **创建自查询检索器**：利用 `from_llm` 方法创建查询构造器和翻译器
3. **执行查询**：用自然语言发起调用，检索器内部执行"构造"和"翻译"两个步骤

### 二、文本到Cypher

**Cypher** 是图数据库（如 Neo4j）中最常用的查询语言，其地位类似于 SQL 之于关系数据库。

**"文本到Cypher"的原理**：
利用大语言模型（LLM）将用户的自然语言问题直接翻译成一句精准的 Cypher 查询语句。

工作流程：
1. 接收用户的自然语言问题
2. LLM 根据预先提供的图谱模式（Schema），将问题转换为 Cypher 查询
3. 在图数据库上执行该查询，获取精确的结构化数据
4. (可选)将查询结果再次交由 LLM，生成通顺的自然语言答案

## 第三节：文本到SQL

### 一、业务挑战

- **"幻觉"问题**：LLM 可能会"想象"出数据库中不存在的表或字段
- **对数据库结构理解不足**：LLM 需要准确理解表的结构、字段的含义以及表与表之间的关联关系
- **处理用户输入的模糊性**：用户的提问可能存在拼写错误或不规范的表达

### 二、优化策略

1. **提供精确的数据库模式**：向LLM提供数据库中相关表的 `CREATE TABLE` 语句
2. **提供少量高质量的示例**：在提示（Prompt）中加入一些"问题-SQL"的示例对
3. **利用RAG增强上下文**：为数据库构建一个专门的"知识库"，包含DDL定义、Q-SQL示例和表描述
4. **错误修正与反思**：在生成SQL后，系统会尝试执行它。如果数据库返回错误，可以将错误信息反馈给LLM

### 三、实现一个简单的Text2SQL框架

#### 3.1 知识库模块

```python
class SimpleKnowledgeBase:
    """知识库"""
    
    def __init__(self, milvus_uri: str = "http://localhost:19530"):
        self.milvus_uri = milvus_uri
        self.client = MilvusClient(uri=milvus_uri)
        self.embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.collection_name = "text2sql_kb"
```

**设计思想**：
1. **统一知识管理**：将DDL定义、Q-SQL示例和表描述三种类型的知识统一存储
2. **语义检索能力**：使用BGE-M3模型进行向量化，支持中英文混合的语义相似度搜索

#### 3.2 SQL生成模块

```python
class SimpleSQLGenerator:
    """简化的SQL生成器"""
    
    def generate_sql(self, user_query: str, knowledge_results: List[Dict[str, Any]]) -> str:
        """生成SQL语句"""
        context = self._build_context(knowledge_results)
        
        prompt = f"""你是一个SQL专家。请根据以下信息将用户问题转换为SQL查询语句。

数据库信息：
{context}

用户问题：{user_query}

要求：
1. 只返回SQL语句，不要包含任何解释
2. 确保SQL语法正确
3. 使用上下文中提供的表名和字段名
4. 如果需要JOIN，请根据表结构进行合理关联

SQL语句："""
```

**关键设计原则**：
1. **上下文驱动**：通过知识库检索结果构建丰富的上下文信息
2. **结构化提示**：明确的任务要求和格式约束
3. **确定性输出**：设置temperature=0确保相同输入产生相同输出

#### 3.3 代理模块

```python
def query(self, user_question: str) -> Dict[str, Any]:
    """执行Text2SQL查询"""
    # 1. 从知识库检索相关信息
    knowledge_results = self.knowledge_base.search(user_question, self.top_k_retrieval)
    
    # 2. 生成SQL语句
    sql = self.sql_generator.generate_sql(user_question, knowledge_results)
    
    # 3. 执行SQL（带重试机制）
    retry_count = 0
    while retry_count < self.max_retry_count:
        success, result = self._execute_sql(sql)
        
        if success:
            return {"success": True, "sql": sql, "results": result}
        else:
            # 尝试修复SQL
            sql = self.sql_generator.fix_sql(sql, result, knowledge_results)
            retry_count += 1
```

**主要查询流程**：
1. 从知识库检索相关信息
2. 生成SQL语句
3. 执行SQL（带重试机制）

## 第四节：查询重构与分发

### 一、查询翻译

#### 1.1 提示工程

通过精心设计的提示词（Prompt），引导 LLM 将用户的原始查询改写得更清晰、更具体，或者转换成一种更利于检索的叙述风格。

**示例**：让 LLM 直接构建出查询指令
```python
prompt = f"""你是一个智能助手，请将用户的问题转换成一个用于排序视频的JSON指令。

你需要识别用户想要排序的字段和排序方向。
- 排序字段必须是 'view_count' (观看次数) 或 'length' (时长) 之一。
- 排序方向必须是 'asc' (升序) 或 'desc' (降序) 之一。

例如:
- '时间最短的视频' 应转换为 {{"sort_by": "length", "order": "asc"}
- '播放量最高的视频' 应转换为 {{"sort_by": "view_count", "order": "desc"}

请根据以下问题生成JSON指令:
原始问题: "{query}"

JSON指令:"""
```

#### 1.2 多查询分解 (Multi-query)

将复杂问题拆分成多个更简单、更具体的子问题。然后，系统分别对每个子问题进行检索，最后将所有检索到的结果合并、去重，形成一个更全面的上下文。

**示例**：
- **原始问题**："在《流浪地球》中，刘慈欣对人工智能和未来社会结构有何看法？"
- **分解后的子问题**：
  1. "《流浪地球》中描述的人工智能技术有哪些？"
  2. "《流浪地球》中描绘的未来社会是怎样的？"
  3. "刘慈欣关于人工智能的观点是什么？"

#### 1.3 退步提示（Step-Back Prompting）

当面对一个细节繁多或过于具体的问题时，引导模型"退后一步"来探寻原始问题背后的通用原理或核心概念。

**工作流程**：
1. **抽象化**：引导 LLM 从用户的原始具体问题中，生成一个更高层次、更概括的"退步问题"
2. **推理**：先获取"退步问题"的答案，然后将这个通用原理作为上下文，再结合原始的具体问题，进行推理并生成最终答案

#### 1.4 假设性文档嵌入 (HyDE)

HyDE 通过一种巧妙的方式来"绕过"查询与文档之间的语义鸿沟问题：

1. **生成**：调用一个生成式 LLM 根据查询生成一个详细的、可能是理想答案的文档
2. **编码**：将这个假设性文档输入到一个对比编码器中，将其转换为一个高维向量嵌入
3. **检索**：使用这个假设性文档的向量，在向量数据库中执行相似性搜索

### 二、查询路由

**查询路由（Query Routing）** 是用于优化复杂 RAG 系统的一项关键技术。当系统接入了多个不同的数据源或具备多种处理能力时，就需要一个"智能调度中心"来分析用户的查询，并动态选择最合适的处理路径。

#### 2.1 应用场景

1. **数据源路由**：根据查询意图，将其路由到不同的知识库
2. **组件路由**：根据问题的复杂性，将其分配给不同的处理组件
3. **提示模板路由**：为不同类型的任务动态选择最优的提示词模板

#### 2.2 实现方法

##### 2.2.1 基于LLM的意图识别

```python
# 第一步：定义分类器
classifier_prompt = ChatPromptTemplate.from_template(
    """根据用户问题中提到的菜品，将其分类为：['川菜', '粤菜', 或 '其他']。
    不要解释你的理由，只返回一个单词的分类结果。
    问题: {question}"""
)
classifier_chain = classifier_prompt | llm | StrOutputParser()

# 第二步：定义路由分支
router_branch = RunnableBranch(
    (lambda x: "川菜" in x["topic"], sichuan_chain),
    (lambda x: "粤菜" in x["topic"], cantonese_chain),
    general_chain  # 默认选项
)

# 第三步：组合完整路由链
full_router_chain = {"topic": classifier_chain, "question": lambda x: x["question"]} | router_branch
```

##### 2.2.2 嵌入相似性路由

```python
def route(info):
    # 1. 对用户查询进行嵌入
    query_embedding = embeddings.embed_query(info["query"])
    
    # 2. 计算与各路由提示的余弦相似度
    similarity_scores = cosine_similarity([query_embedding], route_prompt_embeddings)[0]
    
    # 3. 找到最相似的路由名称
    chosen_route_index = np.argmax(similarity_scores)
    chosen_route_name = route_names[chosen_route_index]
    
    # 4. 获取并调用对应的处理链，返回结果
    chosen_chain = route_map[chosen_route_name]
    return chosen_chain.invoke(info)
```

## 第五节：检索进阶

### 一、重排序 (Re-ranking)

#### 1.1 RRF (Reciprocal Rank Fusion)

已经在混合检索章节中介绍过，是一种简单而有效的**零样本**重排方法，纯粹基于文档在多个不同检索器结果列表中的**排名**来计算最终分数。

#### 1.2 RankLLM / LLM-based Reranker

直接利用大型语言模型本身来进行重排。通过一个精心设计的提示词来实现，该提示词会包含用户的查询和一系列候选文档，然后要求 LLM 以特定格式输出一个排序后的文档列表，并给出每个文档的相关性分数。

#### 1.3 Cross-Encoder 重排

Cross-Encoder 能提供出色的重排精度。它的工作原理是将查询（Query）和每个候选文档（Document）**拼接**成一个单一的输入，然后将这个整体输入到一个预训练的 Transformer 模型中，模型最终会输出一个单一的分数，这个分数直接代表了文档与查询的**相关性**。

#### 1.4 ColBERT 重排

ColBERT（Contextualized Late Interaction over BERT）是一种创新的重排模型，它在 Cross-Encoder 的高精度和双编码器（Bi-Encoder）的高效率之间取得了平衡。采用了一种"**后期交互**"机制。

**工作流程**：
1. **独立编码**：分别为查询（Query）和文档（Document）中的每个 Token 生成上下文相关的嵌入向量
2. **后期交互**：在查询时，模型会计算查询中每个 Token 的向量与文档中每个 Token 向量之间的最大相似度（MaxSim）
3. **分数聚合**：将查询中所有 Token 得到的最大相似度分数相加，得到最终的相关性总分

### 二、压缩 (Compression)

"压缩"技术旨在解决检索到的文档块虽然整体上与查询相关，但可能包含大量无关的"噪音"文本的问题。

#### 2.1 LangChain 的 ContextualCompressionRetriever

LangChain 提供了一个强大的组件 `ContextualCompressionRetriever` 来实现上下文压缩。它像一个包装器，包裹在基础的检索器之上。当基础检索器返回文档后，`ContextualCompressionRetriever` 会使用一个指定的 `DocumentCompressor` 对这些文档进行处理。

内置的 `DocumentCompressor` 类型：
- `LLMChainExtractor`: 利用一个 LLM Chain 来判断并提取出其中与查询相关的部分
- `LLMChainFilter`: 判断整个文档是否与查询相关，如果相关，则保留整个文档；如果不相关，则直接丢弃
- `EmbeddingsFilter`: 计算查询和每个文档的嵌入向量之间的相似度，只保留那些相似度超过预设阈值的文档

#### 2.2 自定义重排器与压缩管道

以 ColBERT 为例，展示如何集成未被官方支持的功能：

```python
class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT重排器"""
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """对文档进行ColBERT重排序"""
        # 编码查询和文档
        query_embeddings = self._encode_query(query)
        doc_embeddings = self._encode_documents(documents)
        
        # 计算ColBERT相似度
        scores = self._calculate_colbert_similarity(query_embeddings, doc_embeddings)
        
        # 排序并返回前5个
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in scored_docs[:5]]
        
        return reranked_docs
```

### 三、校正 (Correcting)

**校正检索（Corrective-RAG, C-RAG）** 是为解决检索系统可能返回不相关、过时或完全错误的文档的问题而提出的一种策略。

C-RAG 的工作流程可以概括为 **"检索-评估-行动"** 三个阶段：

1. **检索 (Retrieve)**：与标准 RAG 一样，首先根据用户查询从知识库中检索一组文档
2. **评估 (Assess)**：一个"检索评估器"会判断每个文档与查询的相关性，并给出"正确"、"不正确"或"模糊"的标签
3. **行动 (Act)**：根据评估结果，系统会进入不同的知识修正与获取流程：
   - **如果评估为"正确"**：系统会进入"知识精炼"环节
   - **如果评估为"不正确"**：系统认为内部知识库无法回答问题，此时会触发"知识搜索"
   - **如果评估为"模糊"**：同样会触发"知识搜索"，但通常会直接使用原始查询进行 Web 搜索

## 学习心得

通过第四章的学习，我对高级检索技术有了全面的认识：

1. **混合检索的优势**：结合稀疏向量的精确匹配和密集向量的语义理解，能够同时利用两种方法的优势，显著提升检索效果

2. **查询构建的重要性**：通过自查询检索器、文本到Cypher、文本到SQL等技术，可以将用户的自然语言查询转换为针对特定数据源的结构化查询语言

3. **查询重构与分发的价值**：通过提示工程、多查询分解、退步提示、HyDE等技术，可以优化用户的原始查询，使其更适合检索

4. **查询路由的智能化**：基于LLM的意图识别和嵌入相似性路由，可以将查询智能分发到最合适的数据源或处理组件

5. **检索进阶技术的必要性**：重排序、压缩和校正等技术解决了传统RAG的固有局限性，能够显著提升检索质量和生成答案的准确性

6. **框架与自定义的平衡**：虽然LangChain、LlamaIndex等框架提供了丰富的功能，但在实际应用中，理解原理并根据需求进行自定义实现往往更灵活、更可控

这些高级检索技术为构建生产级、高质量的RAG系统提供了强有力的工具和方法论。

> 参考教程： https://datawhalechina.github.io/all-in-rag/