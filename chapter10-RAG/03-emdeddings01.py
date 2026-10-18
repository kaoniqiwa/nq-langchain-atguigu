"""
向嵌入处理器提交一个嵌入请求。
"""

import os

from dotenv import load_dotenv
from openrouter import OpenRouter
from openrouter.operations import CreateEmbeddingsResponseBody
from rich import print as rprint
from rich import reconfigure

load_dotenv()
reconfigure(force_terminal=True)

text = "Hello, world"
API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL")

if not API_KEY:
    raise ValueError("请设置 OPENROUTER_API_KEY 环境变量")
if not MODEL:
    raise ValueError("请设置 OPENROUTER_EMBEDDING_MODEL 环境变量")
try:
    with OpenRouter(api_key=API_KEY) as open_router:
        res: CreateEmbeddingsResponseBody | str = open_router.embeddings.generate(
            input=text, model=MODEL
        )
        if isinstance(res, CreateEmbeddingsResponseBody):
            final_res = {}
            for field in CreateEmbeddingsResponseBody.model_fields:
                if field not in res.model_fields_set:
                    continue
                value = getattr(res, field)
                if field == "data":
                    final_res[field] = {
                        "len": len(value),
                        "content": [
                            {
                                "object": getattr(item, "object", "embedding"),
                                "embedding_len": len(getattr(item, "embedding", [])),
                                "embedding_preview": item.embedding[:100]
                                if item.embedding
                                else [],
                            }
                            for item in value
                        ],
                    }

                else:
                    final_res[field] = value
            rprint(final_res)
        elif isinstance(res, str):
            rprint(f"❌ API 返回了非预期格式的错误信息: {res}")
        else:
            rprint(f"❓ 未知的返回类型: {type(res)}")

except Exception as e:  # noqa: BLE001
    rprint(f"💥 请求失败: {type(e).__name__}: {e}")
