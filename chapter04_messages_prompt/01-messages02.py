"""多轮对话的历史消息裁剪。

模型是无状态的，每轮请求都要把完整历史重新发一遍；会话一长，
token 会迅速膨胀，最终超出模型的上下文窗口。所以需要裁剪——
只保留 system 提示词和最近若干轮对话。

本脚本的流程：

1. 连续问 3 个问题，messages 累积成 1 条 system + 3 轮 × 2 条 = 7 条；
2. 用 keep_recent_messages() 裁成「system + 最近 2 轮」，共 5 条；
3. 拿裁剪后的历史追问「我第一个问题问的是什么？」——
   第 1 轮已经被裁掉，模型答不上来，直观看到裁剪的代价。

最后把这一轮的 AIMessage 落盘到 output/01-messages02-<时间戳>.json。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import SecretStr
from rich.console import Console
from utils import keep_recent_messages, save_response

load_dotenv(override=True)

MODEL = os.environ["DEEPSEEK_MODEL"]

API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')

# 标注成 list[BaseMessage]：列表里会混放 System / Human / AI 三种消息，
# 用基类做元素类型，类型检查器才不会因为某个子类的签名不匹配而报警
messages: list[BaseMessage] = [
    SystemMessage('你是 Python 导师'),
    HumanMessage('什么是列表？用一句解释')
]

# 第 1 轮：把模型的回复 append 回列表，下一轮请求就带着这条上下文一起发出去
r1 = chat.invoke(messages)
messages.append(r1)

# 第 2 轮
messages.append(HumanMessage('列表和元组有什么区别？用一句解释'))
r2 = chat.invoke(messages)
messages.append(r2)

# 第 3 轮
messages.append(HumanMessage('什么是字典呢？用一句解释'))
r3 = chat.invoke(messages)
messages.append(r3)

print(f"原始消息数: {len(messages)}")  # 1 条 system + 3 轮 × 2 条 = 7

# 裁剪：system 置顶保留，对话只留最近 2 轮——第 1 轮从这里起就丢了。
# max_pairs 是 keyword-only 参数，必须写成 max_pairs=2。
# 实现细节（按「条数」切而非严格按「轮」切的坑）见 utils.keep_recent_messages 的文档
optimized = keep_recent_messages(messages, max_pairs=2)
print(f"优化后消息数: {len(optimized)}")  # system + 最近 2 轮 × 2 条 = 5
print("保留的内容: system + 最近 2 轮对话")

optimized.append(HumanMessage('我第一个问题问的是什么？'))
response = chat.invoke(optimized)
Console(force_terminal=True).print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-messages02")
print(f"已写入：{out_path}")
