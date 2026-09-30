"""ChatPromptTemplate 支持的五种消息写法。

前三个脚本各自只用了其中一种，这个脚本把它们混在一起，一次看清区别：

    "纯字符串"                       → 默认当作 human 模板，能插变量
    {"role": ..., "content": ...}     → 按 role 决定类型，能插变量
    ("role", "模板")                  → 同上，元组简写
    HumanMessage(content="...")       → 静态消息，不参与变量提取
    HumanMessagePromptTemplate.from_template("...") → 显式模板对象，能插变量

判断标准只有一条：**这条消息会不会被解析成模板**。
只有模板才会被提取 {xxx} 变量、在 invoke 时替换；普通消息对象原样带进结果。

本例的变量表是 ["name", "user_input"]，分别来自第 2 条（system）和最后一条——
中间那条 HumanMessage 是静态的，不贡献任何变量。

最后把响应落盘到 output/03-chatPromptTemplate04-<时间戳>.json。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import SecretStr
from rich.console import Console

from utils import save_response

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])
BASE_DIR = Path(__file__).resolve().parent

chat_prompt_template = ChatPromptTemplate.from_messages([
    "你好，最近怎么样？",  # 字符串默认 HumanMessagePromptTemplate
    {"role": "system", "content": "你是一个友好的AI助手，你的名字叫{name}"},  # SystemMessagePromptTemplate
    ("ai", "我很好，谢谢"),  # 元组形式，等价于 {"role": "ai", "content": ...}
    HumanMessage(content="一句话介绍你自己"),  # 静态消息无法插入变量
    # 显式构造模板对象，和字符串/dict/元组三种形式殊途同归，最终都是 MessagePromptTemplate
    HumanMessagePromptTemplate.from_template("数学问题:{user_input}")
])
print(chat_prompt_template)

# 变量表由模板决定：name 来自第 2 条，user_input 来自最后一条
prompt_value: PromptValue = chat_prompt_template.invoke({"name": "小智", "user_input": "2+3=?"})
print(prompt_value)

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='deepseek',
    extra_body={"thinking": {"type": "enabled"}}  # 开启思考模式
)

response = chat.invoke(prompt_value)
Console(force_terminal=True).print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "03-chatPromptTemplate04")
print(f"已写入：{out_path}")
