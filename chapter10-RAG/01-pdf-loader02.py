from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from utils import save_documents
from rich import reconfigure, print as rprint

reconfigure(force_terminal=True)

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = (BASE_DIR / '../asset/load' / "04-load.pdf").resolve(strict=True)

loader = PyPDFLoader(file_path=FILE_PATH, extraction_mode='plain')
docs = loader.load()

rprint(docs)

out_path = save_documents(docs, "01-pdf-loader01")
rprint(f"已写入：{out_path}")
