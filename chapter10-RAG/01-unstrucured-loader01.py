import langchain_unstructured
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from utils import save_documents
from rich import reconfigure, print as rprint
from langchain_unstructured import UnstructuredLoader

reconfigure(force_terminal=True)

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = (BASE_DIR / '../asset/load' / "04-load.pdf").resolve(strict=True)

loader = UnstructuredLoader(file_path=FILE_PATH)
docs = loader.load()

rprint(docs)
