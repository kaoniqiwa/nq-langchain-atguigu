import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import SecretStr, BaseModel, Field
from rich import reconfigure, print as rprint
from typing import Optional

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


class Person(BaseModel):
    """人物信息"""
    name: str = Field(description='姓名')
    age: Optional[int] = Field(description='年龄', default=1)
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
response = structured_chat.invoke('张三是一名软件工程师')

rprint(response)
print(type(response))  # <class '__main__.Person'>，不是 AIMessage
print(response.name, response.age, response.occupation)
