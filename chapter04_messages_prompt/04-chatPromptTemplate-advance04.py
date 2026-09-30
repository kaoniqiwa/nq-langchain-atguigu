"""MessagesPlaceholder 单独使用 —— 不套 ChatPromptTemplate，直接接模型。

advance03 是把占位符嵌在 ChatPromptTemplate 里；这里更直接：
MessagesPlaceholder 自己就是一个模板，单独调 format_messages() 就能得到消息列表。

流程：

    MessagesPlaceholder("history")     只声明「这个位置要填一段历史」
    .format_messages(history=[...])    填进去，得到 list[BaseMessage]
    chat.invoke(...)                   把消息列表交给模型，拿到回答

和 advance03 相比，少了 system 那一层外壳，模板里只有历史消息本身。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import MessagesPlaceholder
from pydantic import SecretStr
from rich.console import Console

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

# 显式构造，默认 optional=False，所以下面必须给 history
prompt = MessagesPlaceholder("history")

# format_messages 的入参是关键字参数，键名要和上面那个 "history" 对上；
# 值是一个消息列表，元素可以是元组简写，也可以是 HumanMessage / AIMessage 对象
prompt_value = prompt.format_messages(history=[
    ("human", "你好，请问明天的天气如何？"),
    ("ai", "明天天气晴朗"),
    ("human", "后天的天气怎么样？")
])
print(prompt_value)

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='deepseek',
    extra_body={"thinking": {"type": "enabled"}}  # 开启思考模式
)

# format_messages() 返回的是 list[BaseMessage]，三条消息的角色完整保留；
# 交给 chat.invoke()——传消息列表和传 PromptValue 效果一样
response = chat.invoke(prompt_value)
Console(force_terminal=True).print(response)
