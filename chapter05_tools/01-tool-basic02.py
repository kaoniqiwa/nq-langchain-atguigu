"""把工具绑定给模型 —— 由模型决定调不调，调了再由我们执行。

和 01-tool-basic01.py 的区别：

    01       直接调用      自己拿着工具对象 invoke，模型不参与
    本文件    bind_tools   把工具清单告诉模型，由模型判断该不该调

bind_tools 做的事：把工具的名称、描述、参数 schema 一并写进请求里。
模型看到这些信息后，如果认为需要，返回的就不再是回答，
而是一条带 tool_calls 的 AIMessage，内容是「我要调 get_weather，参数传什么」；
如果它觉得不需要（比如你问的是闲聊），就直接回答，tool_calls 是空的。

模型只负责「发起调用」，不会真的执行你的函数 —— 执行得自己来。
好在 BaseTool.invoke() 认得 ToolCall 这种入参：把整个 tool_call 丢给它，
它会自己拆出 args 当工具入参、拿 id 当 tool_call_id，
返回值直接就是一个能塞回消息列表的 ToolMessage，不用手工拼装。

本脚本走完了一整轮：提问 → 模型要调工具 → 本地执行 → 结果回传 → 模型给出回答。
"""

import os

from typing import cast
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import StructuredTool
from pydantic import SecretStr
from rich import reconfigure, print as rprint

reconfigure(force_terminal=True)

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"

    返回:
        天气信息字符串
    """
    return city + "晴天，温度 15°C"


get_weather = cast(StructuredTool, cast(object, get_weather))
chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')

# 绑定工具：把工具清单交给模型。bind_tools 返回的是一个新的 Runnable，
# 原来的 chat 不受影响，还能继续当普通模型用——
# 这是 LangChain 里 bind_* 系列的通用模式（类似 prompt | model 那条链，只是多挂了配置）
chat_with_tools = chat.bind_tools([get_weather])

messages: list[BaseMessage] = [
    HumanMessage('北京今天的天气如何')
]
# 提问。用不用工具不是我们说了算，是模型自己判断：
# 问的是天气，模型应该会回一条带 tool_calls 的消息，而不是直接回答
response: AIMessage = chat_with_tools.invoke(messages)

# 把模型这条回复也存进历史，后面回传工具结果时要连着它一起发
messages.append(response)

# tool_calls 里每一项有三个字段：
#   name  要调哪个工具（对应函数名）
#   args  传什么参数（由 args_schema 约束）
#   id    这次调用的编号，invoke 会拿它当 ToolMessage 的 tool_call_id
# 模型不打算调工具时，tool_calls 是空列表，走 else 分支
if response.tool_calls:
    for tool_call in response.tool_calls:
        if tool_call.get('name') == 'get_weather':
            # 这里直接把整个 tool_call 交给 invoke，不用自己拆 args、拼 ToolMessage：
            # BaseTool 认得 ToolCall 这种入参，会自动做两件事——
            #   1. 取出 args（{'city': '北京'}）当工具入参
            #   2. 拿 id 当 tool_call_id，把执行结果包成 ToolMessage 返回
            # 对照 01-tool-basic01.py：那里传的是普通 dict，没有 id，
            # 返回值就是函数 return 的那个字符串，不会包成 ToolMessage
            toolMessage: ToolMessage = get_weather.invoke(tool_call)
            print('工具调用结果', toolMessage)
            messages.append(toolMessage)
            # 带着工具结果再问一次。这次模型看到结果了，
            # 会停下 tool_calls，给出面向用户的自然语言回答
            response2 = chat_with_tools.invoke(messages)
            rprint(response2)

else:
    print('AI 直接回答', response)
