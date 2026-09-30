"""手工复现 TypedDict 的结构化输出流程 —— 和 01-pydantic-basic02.py 对照着看。

02-typeddict-basic01.py 一行 with_structured_output(MovieTypedDict) 就拿到了 dict，
内部路径比 Pydantic 那版简单：

    1. convert_to_openai_tool(MovieTypedDict)       转成工具 schema
    2. JsonOutputKeyToolsParser(key_name=...)       把 tool_call 的 args 取出来

关键是第 2 步用的解析器不一样。with_structured_output 内部有个 if/else
（langchain_openai/chat_models/base.py:2714）：Pydantic 类走 PydanticToolsParser，
其他类型（含 TypedDict）走 JsonOutputKeyToolsParser。两者差的就在最后一下——

    PydanticToolsParser       还会 Person(**args) 造实例，顺带做校验
    JsonOutputKeyToolsParser  只把 args 原样返回，不做任何校验

所以这里拿到的是 dict，也所以模型填错了不会有任何提示。

本脚本把中间产物（schema 和解析结果）都打印出来，方便对照。
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputKeyToolsParser
from langchain_core.outputs import ChatGeneration
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool
from pydantic import SecretStr
from rich import reconfigure, print as rprint

from utils import save_response


class MovieTypedDict(TypedDict):
    """电影的详细信息"""

    # 字段说明写成 Annotated[类型, "描述"]，第二个位置**必须是字符串**，
    # langchain 会把它取出来当 description（function_calling.py:285，
    # 非字符串会直接报错）——相当于 Pydantic 的 Field(description=...)
    title: Annotated[str, "电影的正式名称，例如《盗梦空间》"]
    year: Annotated[int, "电影的公映年份，使用四位数字表示"]
    director: Annotated[str, "电影导演的全名"]
    # float | int：允许模型给 9 或 9.0，两种都收
    rating: Annotated[float | int, "电影在10分制下的评分，可包含一位小数"]


# 两个转换函数的区别只在外壳：
#   convert_to_openai_tool       {'type': 'function', 'function': {...}}
#   convert_to_openai_function   {'name': ..., 'description': ..., 'parameters': ...}
# 后者就是前者剥掉外面那层，langchain 内部常常两个都要用
oai_tool = convert_to_openai_tool(MovieTypedDict)
oai_function = convert_to_openai_function(MovieTypedDict)

# key_name 必须等于 schema 里的工具名，解析器靠它认出「这次调用的是哪个工具」。
# first_tool_only=True 表示只取第一个结果，返回单个 dict 而不是 list[dict]
parser = JsonOutputKeyToolsParser(key_name=oai_tool['function']['name'], first_tool_only=True)

print(oai_tool)
print(oai_function)

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    # 和 Pydantic 那组一样，要指定 tool_choice，所以得关掉思考模式
    extra_body={"thinking": {"type": "disabled"}}
)

# 绑定工具，并用 tool_choice 强制模型必须调它——和 Pydantic 那版完全一样，
# 传 TypedDict 类和传 Pydantic 类在这里没有区别
chat_tool = chat.bind_tools([MovieTypedDict], tool_choice=oai_function['name'])

# 这次请求模型返回一条 AIMessage：content 为空，信息全在 tool_calls 里的 args
ai_message = chat_tool.invoke('给我介绍下电影《星际穿越》')

# parse_result 的入参是 list[Generation]，所以要把 AIMessage 包一层 ChatGeneration。
# 平时写的 parser.invoke(ai_message)，内部做的也正是这件事——
# 这里展开写，是为了让「解析器到底吃什么」这一步看得见
response = parser.parse_result([ChatGeneration(message=ai_message)])

rprint(response)
print(type(response))  # <class 'dict'>，不是对象——取字段得写 response['title']

# 落盘。response 是个 dict，不是消息——save_response 已经改造过，
# 三种结果都收：BaseMessage（普通对话）、Pydantic 实例、普通 dict（本文件）
out_path = save_response(response, "02-typeddict-basic02")
print(f"已写入：{out_path}")
