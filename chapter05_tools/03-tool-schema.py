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
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import SecretStr, BaseModel, Field
from rich import reconfigure
from langchain.tools import tool

reconfigure(force_terminal=True)

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


class WeatherInput(BaseModel):
    city: str = Field(description="具体城市", default="北京")


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    """
    return city + "晴天，温度 15°C"


oai_function = convert_to_openai_tool(get_weather)
print(oai_function)
