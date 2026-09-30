"""手工复现 with_structured_output 的内部流程 —— 把 function calling 走一遍。

01-pydantic-basic01.py 里一行 with_structured_output(Person) 就拿到了 Person 实例，
它内部其实做了三件事，拆开就是本脚本的代码：

    1. convert_to_openai_tool(Person)              把 Pydantic 模型转成工具 schema
    2. bind_tools([Person], tool_choice="Person")  强制模型必须调这个工具
    3. PydanticToolsParser(tools=[Person])         把 tool_call 的参数还原成 Person

第 3 步说白了就是 Person(**tool_call["args"])。

和第五章的工具调用有个关键区别：**这里没有工具要执行**。
get_weather 是函数，得真的跑一遍；而 Person 是数据模型，不可执行，
它只是借用了工具的 schema 格式来描述「我要什么形状的数据」。
function calling 在这里被当成约束输出格式的手段，不是「让模型干活」的机制。

所以也就没有第二轮请求：工具调用要「执行 → 把结果发回模型 → 模型再回答」，
而结构化输出到「参数解析成 Person」就结束了，后面没有对话。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import SecretStr, BaseModel, Field
from rich import reconfigure, print as rprint

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


class Person(BaseModel):
    """人物信息"""
    # Field 里的 description 会写进 schema 交给模型，
    # 相当于告诉模型「这个字段该填什么」，写清楚能明显提升抽取准确率
    name: str = Field(description='姓名')
    age: int = Field(description='年龄')
    occupation: str = Field(description='职业')


# 第 1 步：把 Pydantic 模型转成工具 schema。
# name 取自类名 "Person"，description 取自类的 docstring「人物信息」，
# parameters 则由三个字段和它们的类型推出来
oai_function = convert_to_openai_tool(Person)

# 输出：
# {'type': 'function', 'function': {'name': 'Person', 'description': '人物信息',
#  'parameters': {'properties': {
#      'name': {'description': '姓名', 'type': 'string'},
#      'age': {'description': '年龄', 'type': 'integer'},
#      'occupation': {'description': '职业', 'type': 'string'}},
#  'required': ['name', 'age', 'occupation'], 'type': 'object'}}}
print(oai_function)

# 没传 model_provider：init_chat_model 会从模型名的前缀推断
# （deepseek-xxx → deepseek），所以这里可以省掉
chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    # 关键在这行：第 2 步要指定 tool_choice，而 DeepSeek 的思考模式不允许，
    # 会返回 400「Thinking mode does not support this tool_choice」。
    # 直接关掉思考模式即可——结构化抽取不需要思维链
    extra_body={"thinking": {"type": "disabled"}}
)

messages: list[BaseMessage] = [
    HumanMessage('张三是一名30岁的软件工程师')
]

# 第 2 步：绑定工具，并用 tool_choice 强制模型必须调它。
# 指定了 tool_choice，模型就没有「要不要调」的选择权，只能把参数填好返回——
# 这正是结构化输出能稳定拿到结果的原因。
# tool_choice 的值从上面 schema 里取 name，而不是写死字符串：
# 内部的 with_structured_output 也是这么取的，将来类名改了这里不会漏改。
# 第一个参数直接传 Pydantic 类本身，和内部的 bind_tools([schema]) 一致
# （传 convert_to_openai_tool 出来的那个 dict 也能跑通，只是绕了一圈）
tool_chat = chat.bind_tools([Person], tool_choice=oai_function['function']['name'])

# 第 3 步：解析器吃的是什么？是 tool_chat.invoke(messages) 返回的那条 AIMessage：
#   AIMessage(content='', tool_calls=[
#       {'name': 'Person',
#        'args': {'name': '张三', 'age': 30, 'occupation': '软件工程师'},
#        'id': 'call_xxx', 'type': 'tool_call'}])
# content 是空的，模型没说一句话，全部信息都在 tool_calls.args 里。
# 解析器要做的就是把 args 还原成 Person，内部干的就是 Person(**args)；
# first_tool_only=True 表示只取第一个结果，返回单个 Person 而不是 list[Person]，
# 这也正是 with_structured_output 返回一个 Person 实例的原因
parser = PydanticToolsParser(tools=[Person], first_tool_only=True)

# 用 | 串成链：这是 Runnable 重载的 __or__，左边的输出直接喂给右边的输入，
# 等价于 parser.invoke(tool_chat.invoke(messages))。
# 写成链式更好读，而且拼出来的结果本身还是个 Runnable
# （.invoke / .batch / .stream / .ainvoke 都还在）。
# 内部的实现 llm | output_parser 用的也是这个写法
response = (tool_chat | parser).invoke(messages)

# 到这里就结束了，没有第二轮请求：结构化输出的目标是「解析出对象」，
# 不是「让模型接着说话」——这一点和第五章的工具调用不同
rprint(response)
print(type(response))  # <class '__main__.Person'>，不是 AIMessage
print(response.name, response.age, response.occupation)

# 如果模型没按 schema 填（比如 age 给了「三十」），
# Pydantic 会在构造 Person 时抛 ValidationError——
# 这里是结构化输出的校验关口，解析器不会替你兜底
