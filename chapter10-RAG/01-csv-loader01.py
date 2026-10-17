from langchain_community.document_loaders import CSVLoader
from pathlib import Path
from utils import save_documents
from rich import reconfigure, print as rprint

reconfigure(force_terminal=True)

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = (BASE_DIR / '../asset/load' / "02-load.csv").resolve(strict=True)

loader = CSVLoader(file_path=FILE_PATH, encoding="utf-8")
docs = loader.load()

rprint(docs)
out_path = save_documents(docs, "01-csv-loader01")
rprint(f"已写入：{out_path}")
