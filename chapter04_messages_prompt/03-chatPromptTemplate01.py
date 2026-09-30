"""ChatPromptTemplate：把「消息列表 + 变量占位符」做成可复用的模板。

手写 messages 时，同一套提示词换个输入就得整段重写；模板把固定部分抽出来，
变量用 {xxx} 占位，每次调用只传变量值，适合做系统提示词这类反复使用的内容。

本脚本的流程：

1. 用 from_messages() 定义模板，四条消息按 system / human / ai 声明各自角色；
2. invoke({...}) 填充变量，得到一个 PromptValue——本质是一份填好的消息列表；
3. PromptValue 直接交给 chat.invoke()，不用自己再拼 messages。

最后把响应落盘到 output/03-chatPromptTemplate01-<时间戳>.json。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from rich.console import Console

from utils import save_response

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])
BASE_DIR = Path(__file__).resolve().parent

# from_messages 支持多种写法：
#   "纯字符串"                     → 默认当成 human 消息模板
#   ("role", "模板")               → 元组形式，按 role 决定消息类型
#   {"role": ..., "content": ...}  → dict 形式，效果同上
#
# 本文件统一用元组写法，四条消息各是什么角色一眼可见。
#
# 关键：模板里的 {xxx} 是变量，invoke 时传字典填充。
# 注意别写成 HumanMessage(content="{user_input}") 这种消息对象——
# 消息对象会被原样当成静态消息，不参与变量提取，占位符不会被替换，
# 模型收到的就是字面量 "{user_input}"。要传模板就得用上面三种形式。
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手，你的名字叫{name}"),
    ("human", "你好，最近怎么样？"),
    ("ai", "我很好，谢谢"),
    ("human", "{user_input}")
])
print('chat_prompt_template:\n', chat_prompt_template)

# 传变量字典填充模板，得到 PromptValue——本质是一份填好内容的消息列表
prompt_value: PromptValue = chat_prompt_template.invoke({"name": "小智", "user_input": "2+3=?"})
print('prompt_value:\n', prompt_value)

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='deepseek',
    extra_body={"thinking": {"type": "enabled"}}  # 开启思考模式
)

# PromptValue 可以直接喂给 chat.invoke()，不用自己再拼 messages
response = chat.invoke(prompt_value)
Console(force_terminal=True).print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "03-chatPromptTemplate01")
print(f"已写入：{out_path}")
