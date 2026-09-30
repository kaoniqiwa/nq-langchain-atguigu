"""ChatPromptTemplate.partial()：预填部分变量，把不变的部分固定下来。

一个模板如果每次调用都有几个变量是固定值（比如固定的角色设定、固定的目标用户），
每次 invoke 都传一遍很啰嗦。partial() 把它们预填进去，返回一个新模板，
之后调用只需要传剩下的变量。

关键语义：**partial() 不修改原模板，而是返回新模板**。

    chat_prompt_template   仍需 role / audience / task 三个变量
    final_template         已被预填 role / audience，只剩 task

本脚本没有调用模型，只是把两次填充结果打印出来对比。
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

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是{role},目标用户是{audience}"),
    ("user", "{task}")
])
print('chat_prompt_template:\n', chat_prompt_template)

# partial() 返回的是新模板，原模板不受影响：
#   chat_prompt_template 仍需 role / audience / task 三个变量
#   final_template 已被预填 role / audience，只剩 task 一个变量
final_template = chat_prompt_template.partial(role="导游", audience="游客")

# 同一个模板换个 task 就是一次新提问，role 和 audience 在 partial 时已固定住
result1 = final_template.invoke({"task": "介绍一下北京的故宫"})
result2 = final_template.invoke({"task": "介绍一下北京的颐和园"})

print(result1)
print(result2)
