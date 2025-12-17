---
title: "Chapter1/Chapter1 解锁RAG"
categories: 技术
tags: ["RAG", "AI", "检索增强生成"]
id: "rag-introduction-guide"
date: 2025-12-16 18:18:18
recommend: true
top: true
---

## 1. 核心概念拆解：为什么要搞 RAG？

### 1.1 本质理解

RAG (检索增强生成) 说白了就是给 LLM 这种“文科生”发了一本**实时更新的参考书**。

- **参数化知识 (Parametric Knowledge)**：模型训练完就固化在权重里的记忆。相当于它的“内隐记忆”，不仅模糊，而且停留在训练截止日期前（Training Cutoff）。
    
- **非参数化知识 (Non-Parametric Knowledge)**：外挂的向量数据库。这是精准的、可插拔的“外部存储”。
    

RAG 的运行逻辑 = 检索 (Retrieval) + 生成 (Generation)

即：用户提问 -> 向量化 -> 数据库捞相关片段 -> 拼接到 Prompt -> 喂给 LLM -> 输出。

### 1.2 技术选型：RAG vs Fine-tuning



|**维度**|**RAG (外挂知识库)**|**Fine-tuning (微调)**|
|---|---|---|
|**知识更新**|**实时** (更新数据库即可)|**滞后** (需重新训练)|
|**数据隐私**|本地闭环，敏感数据不进模型|数据需进入模型权重|
|**幻觉问题**|低 (有参考依据)|中 (仍可能一本正经胡说八道)|
|**适用场景**|查文档、客服、私有知识库|学习特定的语言风格、指令格式|

**结论：** 除非模型根本听不懂你的指令（Capability issue），否则凡是涉及到“知识缺乏”的问题，首选 RAG。

---

## 2. RAG 架构演进

从简单的线性流程到现在复杂的 Agentic 模式，笔记总结了三个阶段：

1. **Naive RAG (初级)**：
    
    - 流程：Retrieve -> Generate。
        
    - _痛点_：检索质量差直接导致回答崩坏（Garbage In, Garbage Out）。
        
2. **Advanced RAG (高级)**：
    
    - 增加了 **Pre-Retrieval** (查询重写，比如把用户的模糊问题改写得更利于检索) 和 **Post-Retrieval** (Rerank 重排序，把相关性最高的排前面)。
        
3. **Modular RAG (模块化)**：
    
    - 支持路由 (Routing)，根据问题难度决定是否检索，或者去哪个库检索。这是目前 Agentic RAG 的雏形。
        

---

## 3. 最小可行性系统 (MVP) 搭建

**工具栈**：LangChain (虽然臃肿但生态好) + DeepSeek API + BGE-Small (Embedding) + 内存向量库。


1. **ETL**：加载文档 -> 清洗 -> 分块 (Chunking)。_分块策略非常关键，直接影响检索粒度。_
    
2. **Embedding**：文本转向量。
    
3. **Indexing**：存入向量库。
    
4. **Retrieval & Generation**：语义搜索 + Prompt 组装。
    

## 4.示例

```
import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

load_dotenv()

markdown_path = r"/home/dorri/.ssh/05_LLMs_STUDY/all-in-rag-main/data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)

```


## 5. 作业

1. LangChain代码最终得到的输出携带了各种参数，查询相关资料尝试把这些参数过滤掉得到`content`里的具体回答。
    
    ```python
   直接访问content属性
   ================================================================================

```angular2html
根据上下文信息，文中举了以下例子：

1.  **自然界中的羚羊**：羚羊出生后通过试错学习站立和奔跑。
2.  **股票交易**：通过不断买卖并根据市场反馈学习如何最大化奖励。
3.  **玩雅达利游戏（如Breakout、Pong）**：通过不断试错学习如何通关。
4.  **AlphaGo**：DeepMind的强化学习算法，能够击败人类顶尖棋手。
5.  **经典控制问题（如CartPole-v0、MountainCar-v0）**：在Gym库中用于测试强化学习算法的简单环 境。
6.  **K-臂赌博机**：一个理论模型，用于说明探索与利用的困境。


```
    
2. 修改Langchain代码中`RecursiveCharacterTextSplitter()`的参数`chunk_size`和`chunk_overlap`，观察输出结果有什么变化。

```

====================================================================================================
测试不同chunk_size和chunk_overlap参数对RAG结果的影响
====================================================================================================



====================================================================================================
配置: 默认参数
chunk_size: 4000, chunk_overlap: 200
====================================================================================================

分块数量: 6
平均块长度: 3804 字符
最小块长度: 3657 字符
最大块长度: 3934 字符
前3个块的长度: [3853, 3737, 3858]

检索到的文档块数量: 3
检索到的总上下文长度: 11578 字符

回答内容:
--------------------------------------------------------------------------------
根据上下文信息，文中举了以下例子：

1.  **自然界中的例子**：羚羊通过试错学习站立和奔跑。
2.  **现实生活中的例子**：股票交易，通过买卖并根据市场反馈学习最大化奖励。
3.  **游戏中的例子**：
    *   雅达利游戏 Breakout（打砖块）。
    *   雅达利游戏 Pong（乒乓球）。
4.  **强化学习取得超人类表现的例子**：DeepMind 的 AlphaGo 击败人类顶尖棋手。
5.  **Gym 库中的经典控制问题环境例子**：
    *   Acrobot（双连杆机器人）
    *   CartPole（车杆平衡）
    *   MountainCar（小车上山）
    *   Taxi-v3（出租车）
6.  **理论模型例子**：$K$-臂赌博机（多臂赌博机），用于说明探索与利用的窘境。
--------------------------------------------------------------------------------

Token使用: 输入=5429, 输出=198, 总计=5627

====================================================================================================


====================================================================================================
配置: 小块+小重叠
chunk_size: 1000, chunk_overlap: 100
====================================================================================================

分块数量: 26
平均块长度: 895 字符
最小块长度: 542 字符
最大块长度: 990 字符
前3个块的长度: [875, 776, 848]

检索到的文档块数量: 3
检索到的总上下文长度: 2907 字符

回答内容:
--------------------------------------------------------------------------------
根据上下文信息，文中举了以下例子：

1. DeepMind研发的走路的智能体，学习在曲折道路上保持平衡并前进。
2. 机械臂抓取不同形状的物体，通过强化学习学到统一的抓取算法。
3. OpenAI的机械臂翻魔方，先在虚拟环境中训练，再应用到真实机械臂。
4. 穿衣服的智能体，训练其完成穿衣服的精细操作，并能抵抗扰动。
5. 探索和利用的例子：选择餐馆（利用已知喜欢的餐馆 vs. 探索新餐馆）、做广告（利用最优策略 vs. 探索新策略）、挖油（在已知地点挖油 vs. 在新地点探索）、玩游戏（利用固定策略 vs. 尝试新招式） 。
6. Gym库中的CartPole-v0游戏示例，展示如何通过代码与环境交互。
--------------------------------------------------------------------------------

Token使用: 输入=1579, 输出=176, 总计=1755

====================================================================================================


====================================================================================================
配置: 中块+大重叠
chunk_size: 2000, chunk_overlap: 400
====================================================================================================

分块数量: 15
平均块长度: 1771 字符
最小块长度: 325 字符
最大块长度: 1988 字符
前3个块的长度: [1897, 1936, 1801]

检索到的文档块数量: 3
检索到的总上下文长度: 5898 字符

回答内容:
--------------------------------------------------------------------------------
根据上下文，文中举了以下例子：

1. DeepMind研发的走路的智能体，学习在曲折道路上保持平衡前进。
2. 机械臂抓取不同形状的物体，通过强化学习学到统一的抓取算法。
3. OpenAI的机械臂翻魔方，先在虚拟环境中训练再应用到真实机械臂。
4. 穿衣服的智能体，训练实现穿衣服功能并能抵抗扰动。
5. 自然界中的羚羊，通过试错学习站立和奔跑。
6. 股票交易，根据市场反馈学习买卖策略。
7. 玩雅达利游戏（如Pong游戏），通过试错学习通关。
8. Gym库中的经典控制问题，如Acrobot、CartPole、MountainCar等环境。
--------------------------------------------------------------------------------

Token使用: 输入=2975, 输出=159, 总计=3134

====================================================================================================


====================================================================================================
配置: 很小块+小重叠
chunk_size: 500, chunk_overlap: 50
====================================================================================================

分块数量: 61
平均块长度: 375 字符
最小块长度: 103 字符
最大块长度: 496 字符
前3个块的长度: [320, 476, 441]

检索到的文档块数量: 3
检索到的总上下文长度: 1357 字符

回答内容:
--------------------------------------------------------------------------------
根据上下文，文中举的例子包括：

1. **强化学习的例子**：
   - DeepMind 研发的走路的智能体（学习在曲折道路上保持平衡）。
   - 机械臂抓取不同形状的物体（通过强化学习学到一个统一的抓取算法）。

2. **探索和利用的例子**：
   - 选择餐馆（利用：去最喜欢的餐馆；探索：尝试新餐馆）。
   - 做广告（利用：采取最优广告策略；探索：尝试新广告策略）。
   - 挖油（利用：在已知地方挖油；探索：在新地方挖油）。
   - 玩游戏（如《街头霸王》，利用：一直用同一策略；探索：尝试新招式）。

3. **强化学习在现实生活中的例子**：
   - 自然界中羚羊学习站立和奔跑。
   - 股票交易（通过市场反馈学习最大化奖励）。
   - 玩雅达利游戏或其他电脑游戏（通过试错学习通关）。
--------------------------------------------------------------------------------

Token使用: 输入=789, 输出=202, 总计=991

====================================================================================================


====================================================================================================
配置: 小块+无重叠
chunk_size: 1000, chunk_overlap: 0
====================================================================================================

分块数量: 26
平均块长度: 861 字符
最小块长度: 536 字符
最大块长度: 996 字符
前3个块的长度: [875, 776, 972]

检索到的文档块数量: 3
检索到的总上下文长度: 2574 字符

回答内容:
--------------------------------------------------------------------------------
根据提供的上下文，文中举了以下例子：

1. DeepMind研发的走路智能体（学习在曲折道路上行走并保持平衡）。
2. 机械臂抓取（使用强化学习训练机械臂抓取不同形状的物体）。
3. OpenAI的机械臂翻魔方（通过虚拟环境训练机械臂玩魔方）。
4. 穿衣服的智能体（训练强化学习智能体完成穿衣服的精细操作）。
5. 自然界中的羚羊（通过试错学习站立和奔跑）。
6. 股票交易（将买卖股票视为强化学习过程）。
7. 玩雅达利游戏（如Pong游戏，通过试错学习通关）。
8. Gym库中的经典控制问题（如Acrobot、CartPole、MountainCar等环境）。

这些例子均用于说明强化学习在不同领域的应用。
--------------------------------------------------------------------------------

Token使用: 输入=1394, 输出=175, 总计=1569

====================================================================================================


总结:
1. chunk_size越小，分块数量越多，每个块包含的上下文越少
2. chunk_overlap越大，相邻块之间的重叠越多，有助于保持上下文的连续性
3. 过小的chunk_size可能导致语义不完整，过大的chunk_size可能导致检索不精确
4. 适当的chunk_overlap可以避免在块边界处丢失重要信息

```


3.给LlamaIndex代码添加代码注释。

```
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'

# 全局模型配置
Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# RAG索引构建
docs = SimpleDirectoryReader(input_files=[r"/home/dorri/.ssh/05_LLMs_STUDY/all-in-rag-main/data/C1/markdown/easy-rl-chapter1.md"]).load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

# 输出
print(query_engine.get_prompts())          # 查看Prompt模板
print(query_engine.query("文中举了哪些例子?")) # 执行查询
```

> 参考： https://datawhalechina.github.io/all-in-rag/


