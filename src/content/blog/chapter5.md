---
title: "Chapter5 格式化生成与查询翻译"
categories: 技术
tags: ["RAG", "格式化生成", "输出解析器", "Pydantic", "函数调用"]
id: "rag-formatted-generation"
date: 2025-12-22 18:18:18
recommend: false
top: true
---

## 第五章学习笔记

## 第一节 格式化生成

### 一、为什么需要格式化生成？

在许多实际应用场景中，从大语言模型（LLM）那里获得一段非结构化的文本往往不满足需求。格式化生成是连接 LLM 的自然语言理解能力和下游应用程序的程序化逻辑之间的关键技术。

**典型应用场景**：
- **RAG 驱动的电商客服**：当用户询问"推荐几款适合程序员的键盘"时，希望 LLM 返回一个包含产品名称、价格、特性和购买链接的 JSON 列表
- **自然语言转 API 调用**：用户说"帮我查一下明天从上海到北京的航班"，系统需要将这句话解析成一个结构化的 API 请求
- **数据自动提取**：从一篇新闻文章中，自动抽取出事件、时间、地点、涉及人物等关键信息，并以结构化形式存入数据库

### 二、格式化生成的实现方法

#### 2.1 Output Parsers

LangChain 提供了一个强大的组件——`OutputParsers`（输出解析器），专门用于处理 LLM 的输出。

**核心思想**：
1. **提供格式指令**：在发送给 LLM 的提示（Prompt）中，自动注入一段关于如何格式化输出的指令
2. **解析模型输出**：接收 LLM 返回的纯文本字符串，并将其解析成预期的结构化数据

**常用解析器类型**：
- **StrOutputParser**：最基础的输出解析器，简单地将 LLM 的输出作为字符串返回
- **JsonOutputParser**：可以解析包含嵌套结构和列表的复杂 JSON 字符串
- **PydanticOutputParser**：通过与 Pydantic 模型结合，可以实现对输出格式最严格的定义和验证

#### 2.2 PydanticOutputParser 工作原理

PydanticOutputParser 是最强大的输出解析器之一，其工作流程如下：

1. **定义数据模型**：使用 Pydantic 的 `BaseModel` 定义期望的数据结构
2. **生成格式指令**：解析器会执行以下操作：
   - 调用 Pydantic 模型的 `.model_json_schema()` 方法，提取出该数据结构的 JSON Schema 定义
   - 对该 Schema 进行简化，并将其嵌入到一个预设的、指导性的提示模板中
3. **构建并执行调用链**：通过 LangChain 表达式语言（LCEL），将 `prompt`、`llm` 和 `parser` 链接起来
4. **解析与验证**：接收 LLM 返回的字符串后，执行两步解析过程：
   - 首先将其解析成一个 Python 字典
   - 然后使用 Pydantic 模型的 `.model_validate()` 方法验证这个字典

**代码示例**：
```python
# 1. 定义期望的数据结构
class PersonInfo(BaseModel):
    """用于存储个人信息的数据结构。"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")

# 2. 基于 Pydantic 模型，创建解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. 创建提示模板，注入格式指令
prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. 创建处理链
chain = prompt | llm | parser

# 5. 执行调用
result = chain.invoke({"text": "张三今年30岁，他擅长Python和Go语言。"})
```

#### 2.3 LlamaIndex 的输出解析

LlamaIndex 的输出解析与生成过程紧密结合，主要体现在两大核心组件中：

1. **响应合成（Response Synthesis）**：
   - 在 RAG 流程中，检索器召回一系列相关的文本块后，响应合成器负责接收这些文本块和原始查询，并以一种更智能的方式将它们呈现给 LLM
   - 例如，它可以逐块处理信息并迭代地优化答案（`refine` 模式），或者将尽可能多的文本块压缩进单次 LLM 调用中（`compact` 模式）

2. **结构化输出**：
   - 当需要 LLM 返回结构化数据（如 JSON）而非纯文本时，LlamaIndex 主要使用 **Pydantic 程序（Pydantic Programs）**
   - 这与 LangChain 的 `PydanticOutputParser` 思想一致：定义 Schema、引导生成、解析验证

#### 2.4 不依赖框架的简单实现思路

如果不依赖特定的框架，也可以通过提示工程（Prompt Engineering）的技巧来实现格式化生成：

**主要技巧**：
- **明确要求 JSON 格式**：在提示中直接、强硬地要求模型"必须返回一个 JSON 对象"
- **提供 JSON Schema**：在提示中给出你想要的 JSON 对象的模式（Schema），描述每个键的含义和数据类型
- **提供 few-shot 示例**：给出 1-2 个"用户输入 -> 期望的 JSON 输出"的完整示例
- **使用语法约束**：对于一些本地部署的开源模型，可以使用 GBNF (GGML BNF) 等语法文件来强制约束模型的输出

### 三、Function Calling

#### 3.1 概念与工作流程

Function Calling（或称 Tool Calling）是近年来 LLM 领域的一个重要进展，提升了模型与外部世界交互和生成结构化数据的能力。

**核心工作流程**：
1. **定义工具**：以特定格式（通常是 JSON Schema）定义好可用的工具，包括工具的名称、功能描述、以及需要的参数
2. **用户提问**：用户发起一个需要调用工具才能回答的请求
3. **模型决策**：模型接收到请求后，分析用户的意图，并匹配最合适的工具。它返回一个包含 `tool_calls` 的特殊响应
4. **代码执行**：应用接收到这个指令，解析出工具名称和参数，然后在代码层面实际执行这个工具
5. **结果反馈**：将工具的执行结果包装成一个 `role` 为 `tool` 的消息，再次发送给模型
6. **最终生成**：模型接收到工具的执行结果后，结合原始问题和工具返回的信息，生成最终的自然语言回答

#### 3.2 Function Calling 实践

```python
# 1. 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                    "type": "string",
                    "description": "城市名称，例如：杭州"
                }
                },
                "required": ["location"]
            }
        }
    }
]

# 2. 用户提问
messages = [{"role": "user", "content": "杭州今天天气怎么样？"}]

# 3. 第一次调用 (User -> Model)
message = send_messages(messages, tools=tools)

# 4. 代码执行：模拟调用天气API，并将结果添加到消息历史
if message.tool_calls:
    tool_call = message.tool_calls[0]
    messages.append(message)  # 添加模型的回复
    tool_output = "24℃，晴朗"  # 模拟API结果
    messages.append({
        "role": "tool", 
        "tool_call_id": tool_call.id, 
        "content": tool_output
    })  # 添加工具执行结果

# 5. 第二次调用 (Tool -> Model)：将工具结果返回给模型，获取最终回答
final_message = send_messages(messages, tools=tools)
print(final_message.content)
```

#### 3.3 Function Calling 的优势

相比于单纯通过提示工程"请求"模型输出 JSON，Function Calling 的优势在于：

- **可靠性更高**：这是模型原生支持的能力，相比于解析可能格式不稳定的纯文本输出，这种方式得到的结构化数据更稳定、更精确
- **意图识别**：它不仅仅是格式化输出，更包含了"意图到函数的映射"。模型能根据用户问题主动选择最合适的工具
- **与外部世界交互**：它是构建能执行实际任务的 AI 代理（Agent）的核心基础，让 LLM 可以查询数据库、调用 API、控制智能家居等

### 四、框架与自定义的平衡

虽然 LangChain、LlamaIndex 等框架提供了丰富的功能，但在实际应用中，理解原理并根据需求进行自定义实现往往更灵活、更可控。

**框架的优势**：
- 快速开发，提供开箱即用的功能
- 经过充分测试，稳定性和可靠性有保障
- 社区支持，文档和示例丰富

**自定义实现的优势**：
- 更灵活，可以根据具体需求定制功能
- 更可控，可以精确控制每个环节的实现细节
- 更轻量，避免引入不必要的依赖和复杂性

**选择建议**：
- 对于原型验证和简单应用，优先使用框架
- 对于生产环境和复杂需求，可以考虑自定义实现或对框架进行扩展
- 无论选择哪种方式，理解底层原理都是最重要的

## 学习心得

通过第五章的学习，我对格式化生成和 Function Calling 有了全面的认识：

1. **格式化生成的重要性**：它是连接 LLM 自然语言理解能力和应用程序程序化逻辑的关键桥梁，在许多实际场景中都是必需的

2. **多种实现方法**：从简单的提示工程到复杂的 Pydantic 解析器，再到 Function Calling，每种方法都有其适用场景和优缺点

3. **Pydantic 的强大之处**：通过数据模型定义、格式指令生成、解析验证的完整流程，实现了对 LLM 输出的严格控制和类型安全

4. **Function Calling 的革命性**：它不仅是格式化输出技术，更是意图识别和工具调用的完整框架，为构建智能 AI 代理奠定了基础

5. **框架与自定义的平衡**：框架提供了快速开发的便利，但自定义实现提供了更大的灵活性和可控性。在实际项目中需要根据具体需求做出权衡

这些技术为构建更智能、更实用的 AI 应用提供了强有力的工具和方法论，是现代 LLM 应用开发不可或缺的组成部分。