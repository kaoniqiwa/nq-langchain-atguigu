"""
返回所有可用嵌入模型及其属性的列表。
"""

import os

from dotenv import load_dotenv
from openrouter import OpenRouter
from rich import print as rprint
from rich import reconfigure

load_dotenv()
reconfigure(force_terminal=True)

API_KEY = os.environ.get("OPENROUTER_API_KEY")


with OpenRouter() as open_router:
    res = open_router.embeddings.list_models(limit=3)
    if res is not None:
        rprint(res.result)
    else:
        rprint("No models found")
