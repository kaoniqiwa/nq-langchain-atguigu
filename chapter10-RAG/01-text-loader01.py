# 本章「文本加载」这几只脚本是同一件事的不同写法，可以对着看：
#   01（本文件） 用 langchain_community 自带的 TextLoader，样本是 UTF-8
#   02          只用 chardet 探测编码，再手动 decode 打印（样本是 GBK）
#   03          chardet 探测 + TextLoader，顺便落盘（样本是 GBK）
#   04          用 utils.py 里手写的 SimpleTextLoader（继承 BaseLoader）
#
# 注意：langchain_community 已进入 sunset（导入时那条 DeprecationWarning 就是它），
# 官方对「纯文本」的建议是直接用内置 file I/O。这里保留自带 loader 是为了对照 API；
# 真实项目里读 txt 用 open() 就够，需要解析 PDF / Word / HTML 时再上 Loader
from langchain_community.document_loaders import TextLoader
from pathlib import Path
from utils import save_documents
from rich import reconfigure, print as rprint

# 强制输出 ANSI 颜色码。rich 默认只在检测到终端时才上色，而 PyCharm 的 Run 窗口
# 常被判定成非终端，不加这行 rprint 就跟普通 print 没区别。
# 代价是输出重定向到文件时会混进转义字符
reconfigure(force_terminal=True)

# 路径以本文件为基准算，不靠运行时的工作目录：__file__ 是脚本自身路径，
# resolve() 把其中的 .. 规范化成绝对路径，于是从哪个目录启动都指向同一个文件
# （这段写法的逐行解释在同目录的 01-text-loader04.py 里）
BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = (BASE_DIR / '../asset/load' / "01-langchain-utf-8.txt").resolve(strict=True)

# 不传 encoding，等于用系统默认编码（macOS 上是 UTF-8）。这个样本正好是 UTF-8，
# 所以读得对；换成 GBK 文件就会乱码或抛 UnicodeDecodeError，
# 那时要么显式传 encoding="gbk"，要么自己探测（见 03 的做法）
loader = TextLoader(file_path=FILE_PATH)
# 返回值一定是 list[Document]，只有一个文件也是列表——BaseLoader 的统一约定，
# 因为 PDF、目录类加载器一个文件能产出很多条
docs = loader.load()

# rprint 是 rich 版的 print：Document 这种长对象会按语法高亮加折行输出，
# 比裸 print 挤在一整行 repr 里好读
rprint(docs)
# 落盘到 output/01-text-loader01-<时间戳>.json，能看清 Document 的真实结构：
# page_content 是正文，metadata 里 source 记着来源文件
# （自带的 TextLoader 只补 source 这一个键；手写的 SimpleTextLoader 还会多一个 type）
out_path = save_documents(docs, "01-text-loader01")
rprint(f"已写入：{out_path}")
