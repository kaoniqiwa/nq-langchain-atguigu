import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import SecretStr, BaseModel, Field
from rich import reconfigure, print as rprint

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


@dataclass
class Person:
    """人物信息"""
    # Field 里的 description 会写进 schema 交给模型，
    # 相当于告诉模型「这个字段该填什么」，写清楚能明显提升抽取准确率
    name: str = Field(description='姓名')
    age: int = Field(description='年龄')
    occupation: str = Field(description='职业')


chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    extra_body={"thinking": {"type": "disabled"}}
)

structured_chat = chat.with_structured_output(Person)

# 一句话里包含三个字段，让模型自己抽出来填进结构
response = structured_chat.invoke('张三是一名30岁的软件工程师')

rprint(response)  # {'name': '张三', 'age': 30, 'occupation': '软件工程师'}

print(type(response))  # <class 'dict'>
