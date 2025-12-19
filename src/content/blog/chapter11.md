---
title: "Chapter11 知识图谱与Neo4j"
categories: 技术
tags: ["知识图谱", "Neo4j", "图数据库", "Cypher", "实体关系"]
id: "knowledge-graph-neo4j"
date: 2025-12-27 18:18:18
recommend: false
top: true
hide: true
---

## 第十一章学习笔记：知识图谱与Neo4j

## 学习目标

1. 理解知识图谱的基本概念和应用场景
2. 掌握知识图谱的构建方法和技术流程
3. 学习Neo4j图数据库的核心概念和基本操作
4. 掌握Cypher查询语言的使用方法
5. 了解Neo4j的安装和配置方法

## 学习内容

### 一、知识图谱基础

#### 1.1 什么是知识图谱

**知识图谱（Knowledge Graph, KG）**源于自然语言理解，其目标是用一种结构化的方式，来描述现实世界中的实体及其相互关系。它主要由两个核心要素构成：

1. **节点（Nodes）**：代表现实世界中的"实体"（Entities），例如一个人、一部电影、一家公司或一个具体概念。
2. **边（Edges）**：代表实体与实体之间的"关系"（Relations）。

这些元素共同构成了一个庞大的语义网络，其基本结构可以表示为**（实体）- [关系] -> （实体）**的三元组（Triples）。例如，"饺子"和"哪吒2"是两个实体，"导演"就是它们之间的关系，构成一个知识三元组：（饺子）- [导演] -> （哪吒2）。

#### 1.2 知识图谱的应用

知识图谱并非一个孤立的学术概念，它在工业界有着广泛且深入的应用，尤其是在需要深度结合领域知识的场景中。

**风险识别与网络分析**：
- **犯罪网络侦查**：公安部门可以利用通话记录、社交关系、转账流水等信息构建犯罪嫌疑人网络。通过分析网络中的核心人物和资金流向，可以有效地打击整个犯罪团伙。
- **信用卡反欺诈**：银行可以将申请人的信息构建成关系网络，识别出"欺诈团伙"，如多个申请人共享同一个联系电话或与已知欺诈分子有紧密的社交关系。

**智能诊断与运维**：
- **工业设备运维**：将设备的各种"故障现象"、"故障原因"、"解决方案"和"所需零件"构建成知识图谱，帮助快速定位问题并给出维修建议。
- **医疗辅助诊断**：构建"病症"、"疾病"、"检查项目"、"治疗方案"、"药品"之间的关系图谱，辅助医生进行诊断和治疗。

**特定领域聊天机器人**：
对于垂直领域，基于知识图谱的问答系统（KBQA）因其答案的准确性和可解释性，具有不可替代的价值。工作流程：
1. **意图识别**：判断用户提问的意图
2. **槽位填充 (实体抽取)**：从问题中抽取出关键信息
3. **知识查询**：利用抽取出的实体在知识图谱中进行精确查询
4. **回复生成**：将查询到的结果通过预设的模板生成自然语言回复

#### 1.3 知识图谱的构建

**经典构建流程**：
传统的知识图谱构建过程主要依赖于两项关键的NLP技术：
1. **命名实体识别 (NER)**：从文本中识别并抽取出特定类别的实体
2. **关系抽取 (RE)**：在识别出实体的基础上，进一步判断实体与实体之间存在何种语义关系

**大模型带来的革新**：
随着大语言模型的兴起，传统的NLP任务流程正在被重塑。LLM同样具备强大的实体识别和关系抽取能力，但呈现出深度融合的趋势。

- **局限性与挑战**：完全依赖LLM会面临成本高昂、数据隐私以及"幻觉"问题
- **融合方案**：微软提出的GraphRAG将知识图谱作为一个可靠、可随时更新的外部知识库，基于图结构进行"子图检索"，而非检索孤立事实

### 二、Neo4j图数据库

#### 2.1 Neo4j核心概念

Neo4j的数据模型主要包含以下几个概念：

- **节点 (Node)**：图中的基本数据单元，用于表示现实世界中的实体。在关系型数据库中，节点可以类比为表中的一行。

- **标签 (Label)**：用于为节点分类或打上"类型"标记。一个节点可以拥有一个或多个标签。

- **关系 (Relationship)**：图数据库的精髓，连接两个节点并明确地定义它们之间的联系。每个关系具有以下特点：
  - **有方向**：关系总是从一个"起始节点"指向一个"结束节点"
  - **有类型**：每个关系都必须有一个类型
  - **可以拥有属性**：关系也可以存储属性

- **属性 (Property)**：以键值对形式存储在节点和关系上的详细信息

这四个概念共同构成了一个灵活而强大的数据模型。

#### 2.2 查询语言：Cypher

Cypher是Neo4j的声明式图形查询语言，语法灵感来源于SQL，但针对图的特性进行了优化。

例如，查找在电影《黑客帝国》中出演过的所有演员：
```cypher
MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie {title: 'The Matrix'})
RETURN actor.name
```

Cypher语法特点：
- 使用ASCII艺术风格来表示图模式
- 支持复杂的模式匹配和路径查询
- 提供丰富的聚合函数和排序功能

#### 2.3 Neo4j安装与使用

**Neo4j Desktop (推荐用于本地学习)**：
1. 访问[Neo4j官网](https://neo4j.com/download/)下载Desktop版本
2. 填写注册表单并下载安装包
3. 双击安装文件进行安装
4. 首次启动时同意许可协议

**Docker (推荐用于服务器部署与跨平台开发)**：
```bash
docker run \
    --name my-neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

参数说明：
- `-p 7474:7474`: 将容器的HTTP端口映射到本机，用于浏览器访问
- `-p 7687:7687`: 将容器的Bolt驱动端口映射到本机，用于代码连接
- `-v $HOME/neo4j/data:/data`: 将数据目录挂载到本机，确保数据持久化
- `--env NEO4J_AUTH=neo4j/password`: 设置数据库的初始用户名和密码

### 三、Neo4j基本使用

#### 3.1 创建并连接数据库

在Neo4j Desktop中创建本地数据库实例的步骤：
1. **创建实例**：在"Local instances"页面点击"Create instance"按钮
2. **配置实例**：为实例命名，选择Neo4j版本，设置密码
3. **启动与连接**：实例创建后自动启动，通过浏览器访问http://127.0.0.1:7474
4. **登录验证**：使用设置的密码连接数据库

#### 3.2 增删查改操作

**场景设定**：
以菜品信息图谱为例，包含以下实体和关系：
- **实体/标签**：`Ingredient`(食材)、`Dish`(菜品)
- **关系**：`(Dish)-[:包含]->(Ingredient)`、`(Dish)-[:主要食材]->(Ingredient)`、`(Dish)-[:调味]->(Ingredient)`

**创建 (CREATE)**：
- 创建节点：`CREATE (pork:Ingredient {name:'猪肉', category:'肉类', origin:'杭州'});`
- 创建关系：`MATCH (d:Dish {name:'鱼香肉丝'}), (i:Ingredient {name:'猪里脊'}) MERGE (d)-[r:主要食材]->(i);`

**查询 (MATCH)**：
- 基本查询：`MATCH (n) RETURN n LIMIT 25;`
- 条件查询：`MATCH (n:Ingredient) WHERE n.name IN ['猪里脊','鸡蛋'] RETURN n;`
- 关联查询：`MATCH (d:Dish)-[:包含]->(i:Ingredient) WHERE d.name IN ['鱼香肉丝', '木须肉'] RETURN d.name AS 菜品, collect(i.name) AS 食材列表;`

**更新 (SET & MERGE)**：
- 更新属性：`MATCH (i:Ingredient {name:'猪肉'}) SET i.is_frozen = true, i.origin = '金华' RETURN i;`
- 插入或更新：`MERGE (n:Ingredient {name: '大蒜'}) ON CREATE SET n.created = timestamp(), n.stock = 100 ON MATCH SET n.stock = coalesce(n.stock, 0) - 1;`

**删除 (DELETE & REMOVE)**：
- 删除属性：`MATCH (i:Ingredient {name:'大蒜'}) REMOVE i.created RETURN i;`
- 删除节点和关系：`MATCH (i:Ingredient {name:'大蒜'}) DETACH DELETE i;`
- 清空数据库：`MATCH (n) DETACH DELETE n;`

#### 3.3 高级操作技巧

**软删除**：
在生产环境中，通过添加状态属性进行软删除，而不是物理删除：
```cypher
MATCH (i:Ingredient {name:'木耳'})
SET i.is_active = false;
```

**批量操作**：
可以在一个查询中创建多个节点和关系：
```cypher
CREATE
    (rousi:Ingredient {name:'猪里脊'}),
    (muer:Ingredient {name:'木耳'}),
    (d1:Dish {name:'鱼香肉丝', cuisine:'川菜'}),
    (d1)-[:包含 {amount:'250g'}]->(rousi),
    (d1)-[:包含]->(muer);
```

**聚合函数**：
使用聚合函数处理多个结果：
```cypher
MATCH (d:Dish)-[:包含]->(i:Ingredient)
WHERE d.name IN ['鱼香肉丝', '木须肉']
RETURN d.name AS 菜品, collect(i.name) AS 食材列表;
```

**排序功能**：
使用ORDER BY对结果进行排序：
```cypher
MATCH (i:Ingredient)
RETURN i.name, i.category
ORDER BY i.name ASC;
```

## 学习总结

本章介绍了知识图谱的基本概念、应用场景和构建方法，以及Neo4j图数据库的核心概念和基本操作。

知识图谱是一种结构化的知识表示方法，通过节点（实体）和边（关系）构成语义网络，广泛应用于风险识别、智能诊断和特定领域聊天机器人等场景。知识图谱的构建主要依赖于命名实体识别和关系抽取两项NLP技术，而大语言模型的出现为这一过程带来了新的可能性。

Neo4j作为流行的图数据库，提供了专门的存储和查询解决方案。其核心概念包括节点、标签、关系和属性，通过Cypher查询语言可以高效地进行图数据操作。Neo4j支持多种安装方式，包括适合本地学习的Desktop版本和适合服务器部署的Docker版本。

在基本操作方面，Neo4j支持完整的CRUD操作：
- CREATE用于创建节点和关系
- MATCH用于查询图数据，支持复杂的模式匹配和条件过滤
- SET和MERGE用于更新属性，MERGE特别适用于"存在则更新，不存在则创建"的场景
- DELETE和REMOVE用于删除数据，DETACH DELETE可以安全地删除节点及其所有关系

通过掌握这些基础知识，我们可以构建和操作复杂的知识图谱，为后续的图RAG系统奠定基础。

> 参考教程： https://datawhalechina.github.io/all-in-rag/