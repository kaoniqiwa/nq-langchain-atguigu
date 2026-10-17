# 这只脚本不碰 LangChain，单独把「编码探测」拎出来演示。
# 起因是：文本文件里存的只是字节，字符长什么样完全取决于用哪张码表去解——
# 同一串 GBK 字节按 UTF-8 解，要么抛 UnicodeDecodeError，要么悄悄给出乱码
# （问号、方块、莫名其妙的汉字）。所以读中文文本，第一步是确认编码。
# 后两只脚本是同一件事的延伸：01-text-loader03.py 把探测结果交给 TextLoader，
# 01-text-loader04.py 用手写的 Loader 绕开编码探测（它只认显式传入的 encoding）
from pathlib import Path
from rich import reconfigure, print as rprint
# chardet 不在 requirements.txt / requirements_full.txt 里，能 import 是因为它是
# langchain-community 的传递依赖。这里既然直接用了它，就该把它显式写进 requirements，
# 否则换个环境就是 ModuleNotFoundError
import chardet

# 强制输出 ANSI 颜色码。rich 默认只在检测到终端时才上色，而 PyCharm 的 Run 窗口
# 常被判定成非终端，不加这行 rprint 就跟普通 print 没区别（下面两处都用的 rprint）。
# 代价是输出重定向到文件时会混进转义字符
reconfigure(force_terminal=True)

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = (BASE_DIR / '../asset/load' / "01-langchain-gbk.txt").resolve(strict=True)

# 用 read_bytes() 而不是 read_text()：探测必须基于原始字节。
# 编码是「字节 → 字符」的映射规则，一旦按错的编码 decode 过，字节就回不来了，
# 再想换张码表重解也没用——这就是必须先拿字节、后定编码的原因
doc_bytes = FILE_PATH.read_bytes()

# detect() 返回四个键：encoding / confidence / language / mime_type。
# 其中 mime_type 是 chardet 7.x 新增的，正是这个新键撞坏了 langchain 的
# autodetect_encoding（完整经过写在 01-text-loader03.py 的文件头）。
# 这里整份打印而不是只取 encoding，是为了能看见 confidence——探测是概率判断，
# 样本越小越不准：实测这个 211 字节的文件，它给的把握只有 0.83
rprint(chardet.detect(
    doc_bytes))  # {'encoding': 'GB18030', 'confidence': 0.834170899663136, 'language': 'zh', 'mime_type': 'text/plain'}

# 这里硬编码 GB18030 而没用上面探测出的 encoding，是有意为之：
# 探测只用来看清结果长什么样，解码则用一个已知可靠的码表，免得被 0.83 的判断带偏。
# GB18030 是 GBK、GB2312 的超集，中文场景当兜底基本不会解错。
# 真要串起来用，写法是 doc_bytes.decode(result['encoding'] or 'gb18030')
# （encoding 为 None 时表示探测失败，得兜底，不能直接塞给 decode）
rprint(doc_bytes.decode("GB18030"))
