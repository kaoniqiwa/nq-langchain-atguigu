"""让模型按指定结构返回数据 —— with_structured_output。

平时模型返回的是一段自然语言，想从里面抠出「姓名 / 年龄 / 职业」还得自己解析。
with_structured_output(Person) 让模型直接吐出符合 Person 结构的结果，
拿到的就是一个 Person 实例，字段点出来就能用。

原理还是 function calling，拆开看就三步（langchain_openai/chat_models/base.py:2699）：

    1. convert_to_openai_tool(schema)["function"]["name"]   取出工具名
    2. bind_tools([Person], tool_choice="Person")           强制模型必须调它
    3. PydanticToolsParser(tools=[Person])                  把 tool_call 里的参数
                                                            解析成 Person 实例

完整的流程图和每一步的源码位置见同目录的 structured-pydantic.md。

第 1 步用的就是 chapter05_tools/02-tool-with-notool.py 里那个函数——
结构化输出和工具调用本来就是同一套机制，只是把「工具」换成了一个数据结构。

注意第 2 步的 tool_choice：DeepSeek 的思考模式不支持强制指定它，会直接报
「Thinking mode does not support this tool_choice」，所以下面关掉了思考模式。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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


chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    # 关键在这行：with_structured_output 会强制模型调用那个「工具」，
    # 而 DeepSeek 的思考模式不允许指定 tool_choice，会返回 400。
    # 直接关掉思考模式即可——结构化抽取不需要思维链
    extra_body={"thinking": {"type": "disabled"}}
)

# 传入 Pydantic 模型，返回的 Runnable 吐出来的就是 Person 实例，不再是自然语言文本
structured_chat = chat.with_structured_output(Person)

# 一句话里包含三个字段，让模型自己抽出来填进结构
response = structured_chat.invoke('张三是一名30岁的软件工程师')

rprint(response)
print(type(response))  # <class '__main__.Person'>，不是 AIMessage
print(response.name, response.age, response.occupation)
