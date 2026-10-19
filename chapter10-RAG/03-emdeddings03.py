import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
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


embeddings_model = OpenAIEmbeddings(model=MODEL, api_key=API_KEY,base_url=BASE_URL)
embed_docs = embeddings_model.embed_query("Hello, world")

rprint(embed_docs[:100])
