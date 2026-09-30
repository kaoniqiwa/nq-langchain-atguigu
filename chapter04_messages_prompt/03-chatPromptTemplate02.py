"""ChatPromptTemplate 的 format() 与 invoke() 的区别。

上一个脚本用 invoke()，拿到的是 PromptValue——一份保留各条消息角色的消息列表；
这个脚本改用 format()，拿到的是纯字符串。

代价在这里：format() 会把四条消息按 "角色: 内容" 拼成一整段文本，形如

    System: 你是一个友好的AI助手，你的名字叫小颜
    Human: 你好，最近怎么样？
    AI: 我很好，谢谢
    Human: 2*6=?

这段文本再交给 chat.invoke() 时，只能作为**一条 HumanMessage** 发出去——
system 和 ai 的角色退化成了文本里的前缀，模型看到的是一段"被转述的对话"，
而不是真正的多条消息。

所以真实项目里应该用 invoke()；format() 主要用于需要把提示词当纯文本
打印查看或做字符串处理的场合。

最后把响应落盘到 output/03-chatPromptTemplate02-<时间戳>.json。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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

# format() 返回的是拼接好的纯字符串（各条消息按 "角色: 内容" 拼成多行文本），
# 类型是 str 而不是 PromptValue——变量名里的 prompt_value 只表示"填好的提示词"
prompt_value: str = chat_prompt_template.format(name='小颜', user_input='2*6=?')
print('prompt_value:\n', prompt_value)

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='deepseek',
    extra_body={"thinking": {"type": "enabled"}}  # 开启思考模式
)

# 传字符串时 LangChain 会把它当成一条 HumanMessage，
# 上面那几个 System / AI 的角色标记就此降级成普通文本
response = chat.invoke(prompt_value)
Console(force_terminal=True).print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "03-chatPromptTemplate02")
print(f"已写入：{out_path}")
