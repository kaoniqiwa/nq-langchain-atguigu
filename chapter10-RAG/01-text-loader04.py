# 手写版的文本加载器，定义在项目根目录的 utils.py 里（SimpleTextLoader），
# 作用就是把 01-text-loader01.py 里 open() + Document 那两步包成一个类，
# 好接进 LangChain 的 Loader 体系——后续切分器、向量库认的是 BaseLoader 的产物。
# 编码默认 utf-8，读 GBK 文件要显式传：SimpleTextLoader("xx.txt", encoding="gbk")。
from pathlib import Path

from utils import SimpleTextLoader, save_documents

# 路径以本文件为基准算，不靠「运行时的工作目录」：
# __file__ 是脚本自身的路径，resolve() 把其中的 .. 规范化成绝对路径，
# 于是从哪个目录启动都指向同一个文件，不像裸的相对路径那样换个工作目录就失效。
# 副作用：metadata["source"] 记的也是这个绝对路径（见打印出来的 Document），
# 换台机器跑，同一个 JSON 里的 source 就不一样了。
BASE_DIR = Path(__file__).resolve().parent
# strict=True：路径写错时在这一行就抛 FileNotFoundError，
# 而不是拖到 loader.load() 里才报错——报错点和写错的地方挨着，好排查
FILE_PATH = (BASE_DIR / '../asset/load' / "01-langchain-utf-8.txt").resolve(strict=True)

# 注意这只解决了「文件在哪」，没解决「utils 能不能 import」：
# from utils import ... 要求项目根在 sys.path 里，PyCharm 会自动加，
# 命令行从项目根目录跑要 PYTHONPATH=. python chapter10-RAG/01-text-loader02.py，
# 否则脚本所在目录（chapter10-RAG）进了 sys.path，根目录没进，报 ModuleNotFoundError
loader = SimpleTextLoader(FILE_PATH)
# 返回值一定是 list[Document]，哪怕文件只有一个、只产出 1 条也是列表。
# 这是 BaseLoader 的统一约定：PDF、目录类加载器一个文件能产出很多条，
# 调用方不该为「只有一条」写特例
docs = loader.load()
print(docs)
# 落盘到 output/01-text-loader02-<时间戳>.json，看得到 Document 的真实结构：
# page_content 是正文，metadata 里 source 记着来源文件、type 记着加载器类型
out_path = save_documents(docs, "01-text-loader02")
print(f"已写入：{out_path}")
