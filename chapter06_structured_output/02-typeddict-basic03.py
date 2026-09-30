"""Annotated 里的 ... 有什么用 —— TypedDict 字段描述的写法。

02-typeddict-basic01.py 里字段写的是 Annotated[str, "描述"]，
这里多了一个 ...。它不是可有可无的摆设，少了它描述会静默失效。

langchain 把 Annotated 的元数据**按位置**塞给 Field
（langchain_core/utils/function_calling.py:287-290）：

    field_kwargs = dict(zip(("default", "description"), annotated_args[1:]))

也就是说：

    元数据第 1 个  →  default
    元数据第 2 个  →  description

所以字符串想当 description，必须先占住 default 位：

    Annotated[str, ..., "描述"]   →  default=...      description="描述"   ✅
    Annotated[str, "描述"]        →  default="描述"   description 没设     ❌

... （Ellipsis）在 pydantic 里表示「必填、无默认值」，正好是我们要的占位符。
第二种写法不会报错，只是描述丢了 —— 模型收不到字段说明，抽取质量会下降，
排查起来还挺费劲（不报错、schema 里少一行而已）。

顺带一提，description 还有第二个来源：类的 Google 风格 docstring
（同文件 298 行的 _parse_google_docstring），两者会合并。
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import SecretStr
from rich import reconfigure, print as rprint


class MovieTypedDict(TypedDict):
    """电影的详细信息"""

    # 第一个 ... 占住 default 位，第二个字符串才是 description。
    # 顺序不能反：反了不报错，只会静默地把描述丢掉
    title: Annotated[str, ..., "电影的正式名称，例如《盗梦空间》"]
    year: Annotated[int, ..., "电影的公映年份，使用四位数字表示"]
    director: Annotated[str, ..., "电影导演的全名"]
    rating: Annotated[float | int, ..., "电影在10分制下的评分，可包含一位小数"]


reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["OPENROUTER_MODEL"]
API_KEY = SecretStr(os.environ["OPENROUTER_API_KEY"])
chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    # 必须显式写：OPENROUTER_MODEL 是 gemini 开头的，
    # 不写会被 init_chat_model 猜成 google_vertexai，见 01-model-openrouter03.py
    model_provider="openrouter"
)

# 和 01 一样走 TypedDict 路径：返回 dict，不做校验（base.py:2714）
structured_chat = chat.with_structured_output(MovieTypedDict)

# 这段提示词故意漏掉了评分：说了 2010 年上映、导演是诺兰，就是没说几分。
# TypedDict 这条路没有校验，模型面对「留空」怎么办全看它自己——
# 可能是 None，也可能干脆不给这个键。用之前先判断一下更稳：response.get('rating')
response = structured_chat.invoke(
    '根据这段话抽取盗梦空间的信息，不包含的信息 可以留空:盗梦空间在2010年上映，导演是克里斯托弗·诺兰。')

rprint(response)
print(type(response))  # <class 'dict'>，不是对象——取字段得写 response['title']
