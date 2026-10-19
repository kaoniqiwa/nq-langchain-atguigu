import os

from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from pydantic import SecretStr
from rich import print as rprint
from rich import reconfigure

load_dotenv()
reconfigure(force_terminal=True)

text = "Hello, world"
API_KEY = SecretStr(os.environ["OPENROUTER_API_KEY"])
MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL")
BASE_URL = os.environ.get("OPENROUTER_BASE_URL")
if not API_KEY:
    raise ValueError("请设置 OPENROUTER_API_KEY 环境变量")
if not MODEL:
    raise ValueError("请设置 OPENROUTER_EMBEDDING_MODEL 环境变量")


"""
provider 为 openai 时，使用 OpenAIEmbeddings 类创建实例
"""
embeddings_model = init_embeddings(
    model=MODEL, provider="openai", base_url=BASE_URL, api_key=API_KEY
)
embed_docs = embeddings_model.embed_query("Hello, world")

rprint(embed_docs[:100])
