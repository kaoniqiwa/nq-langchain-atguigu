from langchain_community.document_loaders import TextLoader
from pathlib import Path
from utils import save_documents
from rich import reconfigure, print as rprint
import chardet

reconfigure(force_terminal=True)

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = (BASE_DIR / '../asset/load' / "01-langchain-gbk.txt").resolve(strict=True)

# 先按二进制读一遍用来探测，下面 TextLoader 又会读一遍，一共两遍 I/O。
# 演示无所谓；真实项目里可以只读一次字节，自己 decode 后直接构造 Document
doc_bytes = FILE_PATH.read_bytes()
result = chardet.detect(doc_bytes)
print(result)

loader = TextLoader(file_path=FILE_PATH, autodetect_encoding=True)
docs = loader.load()

rprint(docs)
out_path = save_documents(docs, "01-text-loader03")
rprint(f"已写入：{out_path}")
