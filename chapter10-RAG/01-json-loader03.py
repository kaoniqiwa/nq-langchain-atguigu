from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from utils import save_documents
from rich import reconfigure, print as rprint

reconfigure(force_terminal=True)

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = (BASE_DIR / '../asset/load' / "03-response.json").resolve(strict=True)

loader = JSONLoader(file_path=FILE_PATH, jq_schema="""
.data.items[] | {
    author,
    created_at,
    content:(.title + "\n" + .content)
}
""", text_content=False)
docs = loader.load()

rprint(docs)

out_path = save_documents(docs, "01-json-loader02")
rprint(f"已写入：{out_path}")
