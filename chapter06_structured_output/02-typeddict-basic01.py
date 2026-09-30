"""用 TypedDict 做结构化输出 —— 不写 Pydantic 也能约束结构。

Pydantic 不是唯一选择，标准库的 TypedDict 也能当 schema 用。
两者写法上的区别只有「字段描述怎么写」：

    Pydantic     name: str = Field(description='姓名')
    TypedDict    name: Annotated[str, '姓名']

描述都会被写进 schema 交给模型，作用一样。

差别大的是**拿到什么**：

    Pydantic     返回模型实例，构造时做校验，字段用 response.name 取
    TypedDict    返回普通 dict，不做任何校验，字段用 response['name'] 取

TypedDict 的代价就在这里：它本来只是给类型检查器看的标注，运行时形同不存在。
模型把 year 填成字符串 "2014"，这里不会有任何报错，得自己检查。
所以需要强校验的场景还是用 Pydantic。
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import SecretStr
from rich import reconfigure, print as rprint


class MovieTypedDict(TypedDict):
    """电影的详细信息"""

    # Annotated 的元数据是**按位置**对应到 Field 参数的
    # （function_calling.py:287-290 用 zip(("default", "description"), 元数据)）：
    #     第 1 个 → default，第 2 个 → description
    # 所以描述想生效，必须先用 ... 把 default 位占住
    # （... 在 pydantic 里表示「必填、无默认值」，正是我们要的）：
    #     Annotated[str, ..., "描述"]   ✅ 描述生效
    #     Annotated[str, "描述"]        ❌ 字符串被当成 default，描述静默丢失
    # 详见 02-typeddict-basic03.py
    title: Annotated[str, ..., "电影的正式名称，例如《盗梦空间》"]
    year: Annotated[int, ..., "电影的公映年份，使用四位数字表示"]
    director: Annotated[str, ..., "电影导演的全名"]
    # float | int：允许模型给 9 或 9.0，两种都收
    rating: Annotated[float | int, ..., "电影在10分制下的评分，可包含一位小数"]


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

# 传入 TypedDict 也能用，但内部走的是另一条解析路径：
# with_structured_output 只对 Pydantic 类用 PydanticToolsParser，
# 其他类型（含 TypedDict）用 JsonOutputKeyToolsParser（base.py:2714）。
# 后者只是把 tool_call 的 args 原样取出来，不做校验——所以下面拿到的是 dict
structured_chat = chat.with_structured_output(MovieTypedDict)

# 只给一部电影名，四个字段全让模型自己补
response = structured_chat.invoke('给我介绍下电影《星际穿越》')
#
rprint(response)
print(type(response))  # <class 'dict'>，不是对象——取字段得写 response['title']
