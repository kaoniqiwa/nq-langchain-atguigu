"""MessagesPlaceholder 的显式写法 —— 和 advance02 的简写对照着看。

advance02 用的是 ("placeholder", "{conversation}") 简写，
这里换成显式构造 MessagesPlaceholder('history')，占的位置一样，
但「必填/可选」的默认行为不同：

    简写 ("placeholder", "{x}")    源码里写死了 optional=True，不传就是空列表
    显式 MessagesPlaceholder("x")  默认 optional=False，**必填**，不传会报错

所以下面 invoke 时必须给 history。

模板结构：system（固定人设）+ history（一整段可变长的消息）。
本脚本只填充模板并打印结果，不调用模型——
要看填充结果怎么喂给模型，见 advance04 / advance05。
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import SecretStr

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "我是一个AI助手"),
    # 显式构造的占位符默认必填，这里占的是「一整段消息列表」的位置，
    # 不是一段文本——所以下面只能给它一个消息列表，不能给字符串
    MessagesPlaceholder('history')
])
print('chat_prompt_template:\n', chat_prompt_template)

# invoke 直接返回 PromptValue，省掉 format() / format_messages() 那一步
# （那两个方法在 03-chatPromptTemplate*.py 里演示过）
# 元组 ("human", ...) 会被转成真正的 HumanMessage / AIMessage，角色完整保留
result = chat_prompt_template.invoke({"history": [
    ("human", "你好，请问明天的天气如何？"),
    ("ai", "明天天气晴朗"),
    ("human", "后天的天气怎么样？")
]})

print(result)
