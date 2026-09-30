"""单独调用一个现成的工具 —— Tavily 联网搜索。

TavilySearch 是官方维护的搜索工具（在 langchain_tavily 包里），
不用自己写 @tool，装好包、给个 key 就能用。

本脚本不接模型，只是手动 invoke 一次，看看工具返回的数据长什么样——
和 chapter05_tools/01-tool-basic01.py 直接调 get_weather 是同一个套路：
先自己把工具试通，再交给 Agent 用（见 01-agent-basic04.py）。

注意返回的是一个 **dict**（Tavily 接口的原始结构，含 query / results 等），
不是 BaseMessage。所以落盘走的是 save_response 里的 Mapping 分支。
"""

import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from rich import reconfigure, print as rprint

from utils import save_response

reconfigure(force_terminal=True)
load_dotenv(override=True)

# TavilySearch 里这个字段声明为必填的 SecretStr（langchain_tavily/_utilities.py:23）：
#     tavily_api_key: SecretStr
# 传 str 进去 pydantic 会自动转成 SecretStr，所以这里不必手动包一层。
# 也可以整个不传——它自己会去读 TAVILY_API_KEY 环境变量
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# max_results=4：最多返回 4 条搜索结果
tool_search = TavilySearch(max_results=4, tavily_api_key=TAVILY_API_KEY)

# 直接调用工具，不经过模型。返回的是 Tavily 接口的原始 dict，大致形如：
#   {"query": "...",
#    "results": [{"title": ..., "url": ..., "content": ...}, ...],
#    ...}
tool_msg = tool_search.invoke("中国江苏省镇江市今天的天气如何")
rprint(tool_msg)

# 落盘。save_response 会递归走一遍 dict，这里的值本来就是 JSON 原生类型，
# 等于原样写出——不像消息对象那样需要转换
out_path = save_response(tool_msg, "01-agent-basic03")
print(f"已写入：{out_path}")
