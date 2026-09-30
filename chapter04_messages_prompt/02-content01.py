"""多模态消息：在一条 HumanMessage 里同时放文字和图片。

content 有两种形态：

    str           纯文本，最常用
    list[dict]    内容块列表（content blocks），每个 dict 是一个块，
                  用来混装文字、图片等不同类型的内容

本例走的是 list 形态，用 OpenAI 风格的 image_url 块：

    {"type": "text", "text": "这张图里有什么？"}
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

url 既可以是公网可访问的 http 地址，也可以是 data URL —— 后者把图片字节直接
内联进请求，省去自己找地方托管图片。image_to_data_uri() 做的就是
「本地图片 → data URL」这步转换。

注意这种块结构是 OpenAI 系的写法，各家的字段名不完全一样，换模型可能要改结构。
对照 02-content02.py：那版用 HumanMessage(content_blocks=[...])，
由 langchain 按目标 provider 负责转换，可移植性更好。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import SecretStr
from rich.console import Console

from utils import image_to_data_uri

load_dotenv(override=True)

MODEL = os.environ["DEEPSEEK_MODEL"]

API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

# 用 __file__ 推导脚本所在目录，避免写死绝对路径，也不受工作目录影响
BASE_DIR = Path(__file__).resolve().parent

# 读本地图片并编码成 data:image/png;base64,.... 的字符串
base64_image = image_to_data_uri(BASE_DIR / "image_test.png")
chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')

# 两条 HumanMessage 放在同一个列表里，正好演示 content 的两种形态可以混用
messages: list[BaseMessage] = [
    # 1. string：content 直接是纯文本，最简单也最常用
    HumanMessage(content="你好"),
    # 2. deepseek 指定的格式,如果用其他模型，以下的结构可能不符合模型规范
    #    content 换成列表，文字块和图片块按顺序排；
    #    模型是按这个顺序理解的——先看到图，再看到「这张图里有什么？」
    HumanMessage(content=[
        {"type": "text", "text": "这张图里有什么？"},
        {"type": "image_url", "image_url": {"url": base64_image}}
    ]),

]

response = chat.invoke(messages)
Console(force_terminal=True).print(response)
