# LangChain Pydantic 结构化输出的工作流程

`with_structured_output(Person)` 一行就能拿到 `Person` 实例，背后走的其实是
function calling。这里把整条链路拆开看。

## 整体流程

一次 `invoke` 走完的路：

```
Person 类（你定义的 Pydantic 模型）
    │
    │ ① convert_to_openai_tool
    ▼
工具 schema   {'type': 'function', 'function': {'name': 'Person', ...}}
    │
    │ ② bind_tools(..., tool_choice='Person')
    ▼
═══════════ 发往 DeepSeek API —— 全程只有这一次请求 ═══════════
    │
    │ 模型按 schema 把参数填好。它返回的不是文字，
    │ 而是一次「调用请求」：我要调 Person，参数是这些
    ▼
AIMessage(content='', tool_calls=[{'name': 'Person', 'args': {...}}])
    │
    │ ③ PydanticToolsParser
    ▼
Person(name='张三', age=30, occupation='软件工程师')
```

链条两头都是你熟悉的 Python 对象，中间这三步全部由 LangChain 完成。

## 三步拆解

| 步骤 | 做什么 | 源码位置（`langchain_openai/chat_models/base.py`） |
| --- | --- | --- |
| ① | `convert_to_openai_tool(schema)["function"]["name"]`，取出工具名 | 2699 |
| ② | `bind_tools([schema], tool_choice=tool_name, ...)`，强制模型必须调它 | 2713 |
| ③ | `PydanticToolsParser(tools=[schema], first_tool_only=True)`，把参数还原成实例 | 2714 |

第 ③ 步的解析器内部干的就是 `Person(**tool_call["args"])`
（`langchain_core/output_parsers/openai_tools.py:371`）；
`first_tool_only=True` 让它返回单个 `Person` 而不是 `list[Person]`，
这正是 `with_structured_output` 返回一个实例的原因。

`ChatDeepSeek` 继承自 `BaseChatOpenAI`（`langchain_deepseek/chat_models.py:47`），
用的就是这份实现。

## 为什么非要绕这一圈

模型 API 只认「自然语言」和「工具 schema」两种输入，没有原生的结构化输出协议。
于是把「我要什么形状的数据」伪装成一个工具，再强制模型去调用它。

这样做的额外好处：`tool_choice` 一指定，模型就没有「要不要调」的选择权，
只能老老实实把参数填成 schema 要求的形状 —— 比在提示词里写「请返回 JSON」可靠得多。

### 一个类比

可以这样理解：**我定义好「参数长什么样」，把一段文字丢给 LLM，
让它把文字里的信息填进这个参数结构**，产出一个能直接展开成实参的字典。

```
'张三是一名30岁的软件工程师'                          ← 输入：一段文字
        │
        │ LLM 按 schema 填参数
        ▼
{'name': '张三', 'age': 30, 'occupation': '软件工程师'}  ← 填好的参数字典
        │
        │ Person(**args)
        ▼
Person(name='张三', age=30, occupation='软件工程师')     ← 实例
```

三点要留意：

- **`Person` 不是函数，是参数规格。** 它有函数签名的形状，但没有函数体。
  对照第五章的 `get_weather` —— 那个才是有签名、有函数体、会被真正执行的完整工具。
- **LLM 产出的是具名参数的键值对（字典），不是实参本身。** 要 `**` 展开之后才谈得上实参。
- **这个类比正好解释了为什么底层是 function calling**：本质上就是「让模型产出一次调用的参数」。
  区别只在于第五章那次调用会被真的执行，而这里的 `Person` 没人去执行它 ——
  参数拿到手就是终点。

## 三个容易误解的点

**1. 只有一次请求。**

和第五章的工具调用不同：那里是「执行工具 → 把结果发回模型 → 模型再回答」，
而结构化输出到「参数解析成 Person」就结束了，后面没有对话。

**2. `Person` 不是可执行的工具。**

它只是借用了工具的 schema 格式来描述数据的形状，本身不可执行。
function calling 在这里是**约束输出格式的手段**，不是「让模型干活」的机制。

**3. 模型的 `content` 是空的。**

因为它的「回答」不是文字，而是一次调用请求，信息全在 `tool_calls` 里。

## DeepSeek 的坑

DeepSeek 的思考模式不支持强制指定 `tool_choice`，会直接报：

```
400 - Thinking mode does not support this tool_choice
```

关掉思考模式即可（结构化抽取不需要思维链）：

```python
chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    extra_body={"thinking": {"type": "disabled"}}
)
```

## 相关文件

- `01-pydantic-basic01.py` —— `with_structured_output` 的基本用法
- `01-pydantic-basic02.py` —— 手工复现上面这条链路，把三步写出来
- `01-pydantic-basic04.py` —— 嵌套结构，一次抽多个对象
- `chapter05_tools/02-tool-with-notool.py` —— `convert_to_openai_tool` 单独演示
