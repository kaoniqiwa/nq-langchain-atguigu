"""ChatPromptTemplate 的嵌套：把子模板整体嵌进外层模板。

前面几个脚本的消息列表里放的都是单条消息，这里放了一个完整的
ChatPromptTemplate——它能像单条消息一样作为外层列表的一项，变量会递归汇总。

用途是把可复用的片段拆开维护：一段固定的系统提示、或一组 few-shot 示例，
单独定义成模板，需要时整体塞进来，不用每次复制那几条消息。

本例的变量表仍然是 ["name", "user_input"]，都来自外层消息——
子模板里没有 {xxx}，只是把「你好，最近怎么样？」这条固定消息包了一层。

最后把响应落盘到 output/03-chatPromptTemplate05-<时间戳>.json。
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

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的AI助手，你的名字叫{name}"),
    # 嵌套：子模板整体作为外层列表的一项，写法上和单条消息没区别。
    # 子模板里若有 {xxx}，同样会被递归汇总到外层的变量表
    ChatPromptTemplate.from_messages([
        ("human", "你好，最近怎么样？"),
    ]),  # 逗号别漏：缺了会被解析成「调用上一个表达式的返回值」，报 TypeError
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
out_path = save_response(response, "03-chatPromptTemplate05")
print(f"已写入：{out_path}")
