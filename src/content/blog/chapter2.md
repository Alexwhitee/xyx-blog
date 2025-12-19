---
title: "Chapter2 数据加载与文本分块"
categories: 技术
tags: ["RAG", "数据加载", "文本分块", "文档处理"]
id: "rag-data-loading-chunking"
date: 2025-12-19 18:18:18
recommend: true
top: true
---

## 第二章学习笔记

## 第一节 数据加载

### 一、文档加载器概述

在RAG系统中，**数据加载**是整个流水线的第一步，也是至关重要的一步。文档加载器负责将各种格式的非结构化文档（如PDF、Word、Markdown、HTML等）转换为程序可以处理的结构化数据。

#### 核心原则
- **"垃圾进，垃圾出 (Garbage In, Garbage Out)"** ——高质量输入是高质量输出的前提
- 数据加载的质量会直接影响后续的索引构建、检索效果和最终的生成质量

#### 主要功能
1. **文档格式解析**：将不同格式的文档解析为文本内容
2. **元数据提取**：提取文档来源、页码等元数据信息
3. **统一数据格式**：将解析后的内容转换为统一的数据格式

### 二、主流RAG文档加载器对比

| 工具名称 | 特点 | 适用场景 | 性能表现 |
|---------|---------|---------|---------|
| **PyMuPDF4LLM** | PDF→Markdown转换，OCR+表格识别 | 科研文献、技术手册 | 开源免费，GPU加速 |
| **TextLoader** | 基础文本文件加载 | 纯文本处理 | 轻量高效 |
| **DirectoryLoader** | 批量目录文件处理 | 混合格式文档库 | 支持多格式扩展 |
| **Unstructured** | 多格式文档解析 | PDF、Word、HTML等 | 统一接口，智能解析 |
| **FireCrawlLoader** | 网页内容抓取 | 在线文档、新闻 | 实时内容获取 |
| **LlamaParse** | 深度PDF结构解析 | 法律合同、学术论文 | 解析精度高，商业API |
| **Docling** | 模块化企业级解析 | 企业合同、报告 | IBM生态兼容 |
| **Marker** | PDF→Markdown，GPU加速 | 科研文献、书籍 | 专注PDF转换 |
| **MinerU** | 多模态集成解析 | 学术文献、财务报表 | 集成LayoutLMv3+YOLOv8 |

### 三、Unstructured文档处理库

[**Unstructured**](https://docs.unstructured.io/open-source/) 是一个专业的文档处理库，专门设计用于RAG和AI微调场景的非结构化数据预处理。

#### 核心优势
1. **格式支持广泛**：支持PDF、Word、Excel、HTML、Markdown等多种格式，提供统一的API接口
2. **智能内容解析**：自动识别文档结构（标题、段落、表格、列表等），保留文档元数据信息

#### 支持的文档元素类型

Unstructured能够识别和分类以下文档元素：

| 元素类型 | 描述 |
|---------|------|
| `Title` | 文档标题 |
| `NarrativeText` | 由多个完整句子组成的正文文本 |
| `ListItem` | 列表项 |
| `Table` | 表格 |
| `Image` | 图像元数据 |
| `Formula` | 公式 |
| `Address` | 物理地址 |
| `EmailAddress` | 邮箱地址 |
| `FigureCaption` | 图片标题/说明文字 |
| `Header` | 文档页眉 |
| `Footer` | 文档页脚 |
| `CodeSnippet` | 代码片段 |
| `PageBreak` | 页面分隔符 |
| `PageNumber` | 页码 |
| `UncategorizedText` | 未分类的自由文本 |
| `CompositeElement` | 分块处理时产生的复合元素 |

### 四、从LangChain封装到原始Unstructured

在第一章的示例中，我们使用了LangChain的`UnstructuredMarkdownLoader`，它是LangChain对Unstructured库的封装。直接使用Unstructured库可以获得更大的灵活性和控制力。

#### partition函数参数解析

- `filename`: 文档文件路径，支持本地文件路径
- `content_type`: 可选参数，指定MIME类型（如"application/pdf"），可绕过自动文件类型检测
- `file`: 可选参数，文件对象，与filename二选一使用
- `url`: 可选参数，远程文档URL，支持直接处理网络文档
- `include_page_breaks`: 布尔值，是否在输出中包含页面分隔符
- `strategy`: 处理策略，可选"auto"、"fast"、"hi_res"等
- `encoding`: 文本编码格式，默认自动检测

#### partition vs partition_pdf

- `partition`函数使用自动文件类型检测，内部会根据文件类型路由到对应的专用函数（如PDF文件会调用`partition_pdf`）
- 如果需要更专业的PDF处理，可以直接使用`from unstructured.partition.pdf import partition_pdf`，它提供更多PDF特有的参数选项，如OCR语言设置、图像提取、表格结构推理等高级功能，同时性能更优

## 练习任务完成汇报

### 练习：使用partition_pdf替换partition函数

#### 任务要求
使用`partition_pdf`替换当前`partition`函数并分别尝试用`hi_res`和`ocr_only`进行解析，观察输出结果有何变化。

#### 实现方案

创建了 `xi-code/exercise4_partition_pdf.py` 文件，实现了以下功能：

1. **导入专用函数**：使用 `from unstructured.partition.pdf import partition_pdf` 替代通用的 `partition` 函数

2. **两种策略对比**：
   - **hi_res策略**：高分辨率策略，使用深度学习模型进行更精确的文档结构识别
   - **ocr_only策略**：仅OCR策略，完全依赖OCR技术提取文本

3. **对比分析**：
   - 统计两种策略解析出的元素数量和字符总数
   - 对比元素类型分布
   - 显示前5个元素示例

#### 实际运行结果与对比

本次在 `rag.pdf` 上分别使用 `hi_res` 和 `ocr_only` 两种策略进行解析，得到的**实际输出关键结果**如下：

- **ocr_only 策略整体统计：**
  - 解析完成: **137 个元素，8282 个字符**
  - 元素类型分布：`{'UncategorizedText': 54, 'Title': 55, 'NarrativeText': 27, 'ListItem': 1}`
  - 前 5 个元素示例（可以明显看到是 OCR 后的“乱码式”英文/符号组合）：
    - Element 1 (`UncategorizedText`): `Bh fe Se «6 Be BR 8H Me OE 6B CR OBS ESR SR ith ...`
    - Element 2 (`Title`): `0 0 Bai @ Bil | eme22n x SME | mm`
    - Element 3 (`NarrativeText`): 一大段混合字母、符号的长串文本（说明 OCR 在该 PDF 上识别质量一般）
    - Element 4 (`UncategorizedText`): `ABB AWARAZ—`
    - Element 5 (`Title`): `MoingSh`

- **两种策略的整体对比：**
  - hi_res 策略: **221 个元素，8265 个字符**
  - ocr_only 策略: **137 个元素，8282 个字符**
  - 元素数量差异: **221 - 137 = 84**
  - 字符数量差异: **8265 - 8282 = -17**（也就是 ocr_only 字符略多一点）

- **元素类型分布对比：**
  - hi_res 策略：
    - UncategorizedText: 85
    - NarrativeText: 68
    - Title: 30
    - Image: 22
    - Header: 4
    - Table: 4
    - FigureCaption: 4
    - ListItem: 4
  - ocr_only 策略：
    - Title: 55
    - UncategorizedText: 54
    - NarrativeText: 27
    - ListItem: 1

#### 从结果中得到的观察与理解

1. **结构识别能力差异明显**
   - `hi_res` 能识别出 `Image`、`Header`、`Table`、`FigureCaption` 等多种结构化元素，说明它不仅做文字识别，还做了版面/布局分析。
   - `ocr_only` 几乎看不到这些结构化类别，主要集中在 `Title`、`UncategorizedText` 和少量 `NarrativeText`，更像是“纯 OCR 把字读出来”，对排版结构理解有限。

2. **元素数量 vs 字符数量**
   - `hi_res` 元素数更多（221 > 137），但字符数略少（8265 < 8282），说明它**把同样的内容拆分成了更多、更细的结构化块**，而不是简单地堆长文本。
   - `ocr_only` 元素更少但字符略多，更像是“粗糙的长段文本堆叠”，结构粒度不够细。

3. **文本质量与可读性**
   - 从 ocr_only 的前 5 个元素可以看到，文本中夹杂大量错误字符和奇怪的单词，说明在这个 PDF 上，OCR 的识别质量有限，**直接用 ocr_only 结果做 RAG，召回质量可能会比较差**。
   - hi_res 由于结合了版面分析和更智能的解析，整体结构更丰富，更适合作为后续分块和索引的输入。

4. **策略选择的实践经验**
   - 对于像 `rag.pdf` 这样**原本就有文本层**的技术文档，`hi_res` 更合适：结构清晰、元素多样，便于后续按标题/小节做精细分块。
   - `ocr_only` 更适合**纯扫描件/图片型 PDF**，在当前这个文档上主要是“强行 OCR”，既慢又不一定准。
   - 这次实验验证了文档中的观点：**在有条件的情况下，应尽量使用结构感知更强的策略（如 hi_res），而不是一味依赖 OCR。**

#### 代码实现

```python
from unstructured.partition.pdf import partition_pdf

# hi_res策略
elements_hi_res = partition_pdf(
    filename=pdf_path,
    strategy="hi_res"
)

# ocr_only策略
elements_ocr_only = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only"
)
```



#### 学习收获

1. **专用函数 vs 通用函数**：`partition_pdf` 相比 `partition` 提供了更多 PDF 特有的参数选项，且在启用 `hi_res` 时能显著提升结构识别能力。
2. **策略选择的重要性**：同一份 `rag.pdf`，`hi_res` 和 `ocr_only` 在元素类型和数量上的差异非常明显，直接影响后续分块与检索质量。
3. **性能与质量的权衡**：`hi_res` 需要加载布局模型 + OCR，速度更慢，但结构更丰富；`ocr_only` 更像“暴力 OCR”，在有文本层的 PDF 上反而不占优势。
4. **实践层面的策略建议**：
   - 对于**文本型/电子版 PDF**（有文本层），优先选择 `hi_res` 或其他结构感知强的策略；
   - 对于**扫描件/图片 PDF**，可以考虑 `ocr_only` 或其他专门的 OCR 流程；
   - 在真实项目中，可以先对小样本做类似本次的对比实验，再为整个文档库确定统一的数据加载策略。


## 第二节 文本分块

### 一、为什么一定要做文本分块

学习完这一节，我最大的感受是：**分块本质上是在做“信息抽样”与“语义打包”**，决定了后面 Embedding、检索和 LLM 看到的到底是哪一部分内容。

- **上下文窗口的硬限制**：
  - 嵌入模型（如 `bge-small-zh-v1.5`）通常只有几百到几千 token 的输入上限，如果原始文档直接丢进去，一定会被截断。
  - LLM 的上下文虽然更长，但检索出来的所有 chunk + 问题 + 系统提示必须一起塞进窗口里，所以单个块不能太大。
- **语义表示的“稀释效应”**：
  - 一个 chunk 最终会被压缩成一个向量，块越长，向量需要“概括”的语义越多，**越难精准代表其中某个细节**。
  - 太大的块会让 embedding 变成“泛泛而谈的摘要”，检索匹配度下降。
- **Lost in the Middle 问题**：
  - 即使 LLM 窗口足够长，把很多大块塞进去，模型更关注开头和结尾，中间的信息反而容易被忽略。
  - 这让我意识到：**不是“给得越多越好”，而是“给得越精准越好”**。

一个直观的记忆：**块太小 → 语义碎片化；块太大 → 语义稀释 + 检索困难，好的 chunk 需要在“完整性”和“专一性”之间取得平衡。**

### 二、基础分块策略理解

#### 1. 固定大小分块（CharacterTextSplitter）

- 文档中的解释让我意识到，LangChain 的固定大小分块其实是“**段落感知的自适应分块**”，并不是真的暴力按字符数切：
  - 先按 `\n\n` 等分隔符切成段落；
  - 再通过 `_merge_splits` 把段落合并到接近 `chunk_size`；
  - 只有在必要时才会出现“单段超长块”。
- **优点**：实现简单、速度快、开销小，适合日志、简单文本预处理等场景。
- **缺点**：语义边界不够精细，可能在句子/话题中间截断，语义连贯性一般。
- 我的理解：如果只是“先把数据跑起来看效果”，可以用它做 baseline，但要做高质量 RAG，后面一般都会换更智能的分块方式。

#### 2. 递归字符分块（RecursiveCharacterTextSplitter）

这一节给我一个很清晰的“分层刀法”印象：

- **核心思想**：
  - 给出一组有优先级的分隔符（如：段落→换行→句号→逗号→空格→字符）；
  - 从优先级最高的开始切，遇到仍然超长的片段，就用下一个更细的分隔符继续切；
  - 分隔符用完还超长，就接受为“大块特例”。
- **相对固定分块的提升**：
  - 尽量在“自然语义边界”（段落、句子）处切断；
  - 遇到极端长句子/段落时，才退化到更细的切法。
- **对中文/代码的支持**：
  - 文档中特别提到可以为中文增加全角逗号、句号等分隔符；
  - 对代码可以用 `from_language`，按照函数、类等结构来分。

我的总结：**递归字符分块 = 保留语义结构的“多级刀法”，是目前通用 RAG 项目里最常用、性价比最高的分块方式之一。**

#### 3. 语义分块（SemanticChunker）

相比前两种“按字符/符号分”，语义分块更像是在做**自动“按话题断句”**：

- 先按句子切，再用嵌入模型算相邻句子之间的语义距离；
- 距离突然变大（语义跳跃明显）的地方，就是“断点候选”；
- 可以用百分位、标准差、四分位距、梯度等统计方法来自动选阈值。

我自己的理解和取舍：

- **优势**：块内部主题高度一致，特别适合法律、科研等长篇、话题切换明显的文本。
- **代价**：需要为大量句子做 embedding + 统计，预处理成本更高；
- **适用场景**：
  - 知识库规模不算特别大，但质量要求很高时（比如内部制度、合约条款库）；
  - 或者只对少量核心文档用语义分块做“精细索引”。

### 三、基于文档结构与其他框架的分块

#### 1. 基于 Markdown / HTML 结构分块

- 使用 `MarkdownHeaderTextSplitter` 等工具，可以**沿着标题层级来切分**，每个块都带着类似：
  - `{"Header 1": "第二章 文本分块", "Header 2": "3.2 递归字符分块"}` 这样的元数据。
- 真正好用的点在于：
  - 可以先按“章节”分成逻辑大块；
  - 再在每个大块内部用 `RecursiveCharacterTextSplitter` 做二次分块；
  - 最终的小块既有合适大小，又保留了完整的“章节路径”，对检索排序和答案引用非常友好。

#### 2. Unstructured 与 LlamaIndex 的分块思路

- **Unstructured**：先做文档元素级的解析（Title、NarrativeText、Table 等），再在元素列表上做 `basic` 或 `by_title` 分块，强调“先理解，再分块”。
- **LlamaIndex**：一切都是 Node，分块只是 Node Parser 里的一个环节，可以组合结构型 + 语义型解析器，并通过元数据维护节点之间的上下文关系。

我的感受是：在简单项目里用 LangChain 自带的分块足够，但一旦文档结构复杂、后续要做多级索引或多模态扩展时，**Unstructured/LlamaIndex 这类“元素/节点优先”的设计思路会更有扩展空间。**

### 四、这一节的整体收获

1. **分块不是机械的“切块”，而是对“知识粒度”的设计**，会直接体现在检索相关性和生成质量上。
2. **没有“一刀切”的最佳参数**：chunk_size 和 chunk_overlap 需要结合文档类型、模型上下文、问答场景反复实验、可视化和调优。
3. **分块策略是可以分层组合的**：结构分块 + 递归字符分块 + 语义分块 / 节点解析，可以一步步从“能用”进化到“好用”。
4. 后面在做向量索引和检索优化时，再回头看这一节，会更清楚：很多"召回不准"和"答案跑题"的问题，其实源头都在这里的分块设计。

> 参考教程： https://datawhalechina.github.io/all-in-rag/ 