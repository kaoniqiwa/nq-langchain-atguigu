"""不用 @tool 装饰器，照样能把函数交给模型当工具用。

@tool 其实做了两件事：

    1. 把函数的签名和 docstring 转成 JSON Schema —— 也就是 function calling
       协议要求的格式，模型靠它知道「这个工具叫什么、收哪些参数」
    2. 把函数包成 BaseTool 对象，让它在本地能 .invoke()、能被 Agent 调用

convert_to_openai_tool 只做第 1 步：输入一个普通函数，输出一个 dict
（OpenAI 风格的 function 定义）。这个 dict 可以直接交给 bind_tools。

所以不写 @tool 也能让模型用上工具，区别只在本地拿到的是什么：

    @tool 装饰              拿到 BaseTool 对象，本地想调就调
    convert_to_openai_tool  拿到一个 dict，本地要执行得按 name 自己找函数

本脚本先看转换出来的 schema 长什么样，再把普通函数直接绑给模型问一次——
bind_tools 不挑食：BaseTool、dict、普通函数它都收。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import SecretStr
from rich import reconfigure, print as rprint

reconfigure(force_terminal=True)

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    Args:
        city: 城市名称，如"北京"、"上海"

    Returns:
        天气信息字符串
    """
    return city + "晴天，温度 15°C"


# 注意这里是个普通函数，没有 @tool。
# 转换的信息全部来自函数本身：函数名 → name，docstring → description，
# 类型注解 → parameters
oai_function = convert_to_openai_tool(get_weather)

# 转换结果：
#
# {
#     'type': 'function',
#     'function': {
#         'name': 'get_weather',
#         'description': '获取指定城市的天气信息 参数:\n    city:
# 城市名称，如"北京"、"上海" 返回:\n    天气信息字符串',
#         'parameters': {
#             'properties': {'city': {'type': 'string'}},
#             'required': ['city'],
#             'type': 'object'
#         }
#     }
# }
#
# 两个值得注意的点：
#   description 是把整个 docstring 原样塞进去的，连「参数:」「返回:」这些
#     中文段落也一并带上——模型只看到这一段，所以工具函数的 docstring
#     要写得简洁清楚，它就是给模型看的说明书
#   parameters 由 city: str 推出来：一个 object，里面一个 string 属性；
#     city 进 required，因为没给默认值
rprint(oai_function)

# 接模型。这里喂给 bind_tools 的是没装饰过的普通函数 get_weather，一样能用——
# bind_tools 既能吃 BaseTool，也能吃上面那种 dict，或者直接给普通函数。
# 不写 @tool 的代价只体现在本地：拿到 tool_calls 之后要执行，
# 得自己按 name 找函数、按 args 传参，没有现成的 .invoke(tool_call) 可用
chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')
chat_with_tools = chat.bind_tools([get_weather])
messages: list[BaseMessage] = [
    HumanMessage('北京今天的天气如何')
]
response = chat_with_tools.invoke(messages)
rprint(response)
