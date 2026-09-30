"""ChatPromptTemplate 的 format_messages()：直接拿到填充好的消息列表。

这是第三个变体，和前面两个对照着看：

    01  invoke()           → PromptValue，包装对象，保留消息结构
    02  format()           → str，拼成一段文本，角色降级成文字前缀
    03  format_messages()  → list[BaseMessage]，裸的消息列表

format_messages() 等价于 invoke().to_messages()。它是三者中最"看得见结构"的：
打印出来能直接看到每条消息的类型和内容，交给 chat.invoke() 又和 01 完全等价。

所以日常用 invoke() 就行，format_messages() 的优势在调试——
想检查"模板到底渲染成了哪几条消息"时，打印它比打印 PromptValue 直观。

最后把响应落盘到 output/03-chatPromptTemplate03-<时间戳>.json。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from rich.console import Console

from utils import save_response

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])
BASE_DIR = Path(__file__).resolve().parent

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手，你的名字叫{name}"),
    ("human", "你好，最近怎么样？"),
    ("ai", "我很好，谢谢"),
    ("human", "{user_input}")
])
print('chat_prompt_template:\n', chat_prompt_template)

# format_messages() 返回 list[BaseMessage]：四条消息各是各的对象，
# 角色信息完整保留，不像 format() 那样被拼成一整段文本
prompt_value: list[BaseMessage] = chat_prompt_template.format_messages(name='小颜', user_input='2*6=?')
print('prompt_value:\n', prompt_value)

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='deepseek',
    extra_body={"thinking": {"type": "enabled"}}  # 开启思考模式
)

# 传消息列表和传 PromptValue 效果一致，都是标准用法
response = chat.invoke(prompt_value)
Console(force_terminal=True).print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "03-chatPromptTemplate03")
print(f"已写入：{out_path}")
