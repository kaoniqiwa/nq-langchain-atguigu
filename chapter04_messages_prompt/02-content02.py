"""多模态消息（二）：用 langchain 标准的 content_blocks 传图片。

和 02-content01.py 的区别只在「图片块怎么写」：

    02-content01   content=[{"type": "image_url", "image_url": {"url": data_url}}]
                   供应商（OpenAI 系）的字段名，url 里还要自己拼 data: 前缀
    本文件          content_blocks=[{"type": "image", "base64": ..., "mime_type": ...}]
                   base64 和 mime_type 是两个独立字段，
                   由 langchain 按目标 provider 负责拼成它认识的格式

所以 content_blocks 是可移植性更好的写法，换模型时更不容易出问题。
两者是同一份内容的两种表达方式，一般二选一，不要同时用。

对应的工具函数也不同：01 用 image_to_data_uri（返回拼好的字符串），
本文件用 image_to_base64（返回拆开的 base64 + mime_type 两个字段）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import SecretStr
from rich.console import Console

from utils import image_to_base64

load_dotenv(override=True)

MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])
# 用 __file__ 推导脚本所在目录，避免写死绝对路径，也不受工作目录影响
BASE_DIR = Path(__file__).resolve().parent

# 返回 {"base64": "...", "mime_type": "image/jpeg"}，
# 字段名和下面的 image 块完全对齐，所以可以直接 ** 展开进去
image_result = image_to_base64(BASE_DIR / "flowers.jpg")

chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')

messages: list[BaseMessage] = [
    # 3. langchain 的标准 content blocks
    #    注意这里用的是 content_blocks= 而不是 content=，
    #    图片块里的 base64 不带 "data:image/jpeg;base64," 前缀，也无需自己拼
    HumanMessage(content_blocks=[
        {"type": "text", "text": "下面几张图里有什么？"},
        # image 是 langchain 内置的 KNOWN_BLOCK_TYPES,而 image_url 是供应商的字段
        # **image_result 把 {"base64": ..., "mime_type": ...} 两个键展开进这个块
        {"type": "image", **image_result}
    ]),

]

response = chat.invoke(messages)
Console(force_terminal=True).print(response)
