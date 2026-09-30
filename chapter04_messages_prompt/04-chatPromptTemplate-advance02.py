"""MessagesPlaceholder：给「一整段消息列表」在模板里留位置。

前面几个模板里，每条消息都是写死的；这里用 ("placeholder", "{变量名}")
占住一个位置，调用时再把一整段消息填进去——最典型的用途是插入会话历史。

填进去的内容会被逐条转成消息对象，可以混用多种写法：
    ("human", "你好")      元组
    HumanMessage("你好")   消息对象
    "你好"                 纯字符串，默认当成 human

一个容易忽略的细节：用 ("placeholder", "{x}") 简写创建出来的占位符是
**可选**的（源码里写死了 optional=True），不传就是空列表；而显式写
MessagesPlaceholder("x") 默认是**必填**的，不传会报错。
需要「有历史就插入、没有就跳过」时用前者。

本脚本没有调用模型，只打印填充结果。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])
BASE_DIR = Path(__file__).resolve().parent

# ("placeholder", "{conversation}") 是 MessagesPlaceholder 的简写：
# 它占的是一整段消息的位置，而不是一段文本
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "我是一个AI助手"),
    ("placeholder", "{conversation}")
])
print('chat_prompt_template:\n', chat_prompt_template)

# 传进去的是一个消息列表，每项会被转成消息对象；
# 这里用元组简写，也可以传 HumanMessage/AIMessage 对象或纯字符串
result = chat_prompt_template.invoke({"conversation": [
    ("human", "你好，请问明天的天气如何？"),
    ("ai", "明天天气晴朗"),
    ("human", "后天的天气怎么样？")
]})

print(result)
