# 切分（splitter）是 RAG 流程的第一步：整篇文档塞不进模型的上下文窗口，
# 而且检索时粒度越细命中越准——按段落/句子切开，命中一段就够；
# 整篇丢进去，检索到的是一大坨，噪声多、还白烧 token。
# 所以「加载 → 切分 → 向量化 → 检索 → 生成」里，切分直接决定后面检索的质量
from rich import reconfigure, print as rprint
from langchain_text_splitters import CharacterTextSplitter

reconfigure(force_terminal=True)

# 三引号里的首尾换行也会算进 text，
# 不过 CharacterTextSplitter 默认 strip_whitespace=True，会把每个片段两端的空白去掉
text = """
LangChain 是一个用于构建基于大语言模型（LLM）应用的开发框架，旨在帮助开发者更高效地集成、管理和增强大语言模型的能力，构建端到端的应用程序。它提供了一套模块化工具和接口，支持从简单的文本生成到复杂的多步骤推理任务
"""

rprint(repr(text))
# CharacterTextSplitter 的工作方式是「先分后合」：
# 先按 separator 把文本切成小片段，再把片段贪心地拼到 chunk_size 以内。
# 所以真正决定切分位置的是 separator，chunk_size 只是合并时的目标长度（见下）
splitter = CharacterTextSplitter(
    # 每个 chunk 的目标长度，单位是字符、不是 token：中文里 1 个汉字算 1 个字符，
    # 而模型上下文上限是按 token 算的（英文大约 4 个字符 ≈ 1 个 token），
    # 两套量纲不同，别拿 chunk_size 直接去比模型的 token 上限
    chunk_size=50,
    # 相邻 chunk 重叠 5 个字符：防止答案正好被切在边界上、两边各留一半都不完整。
    # 代价是内容有重复，检索时可能召回两条高度相似的 chunk
    chunk_overlap=5,
    # 传空字符串 = 不按任何字符分隔，直接逐字符切，再合并到 chunk_size 以内。
    # 这是让 chunk_size 在中文上真正生效的关键——中文句子不用空格断句，
    # 若按 ' ' 切，整段会被当成一个片段放行，chunk_size 就成了摆设（对比见下）
    separator='',
    # 每个 chunk 两端的空白（含换行）会被去掉，免得 chunk 以 \n 开头。
    # 这本就是默认值 True，显式写出来只是让行为可见
    strip_whitespace=True,
)

# 注意 chunk_size 不是硬上限：某个片段本身就超过 chunk_size、中间又没有 separator 可切时，
# 它会原样放行——既不在中间硬切，也不报警告。所以 separator 选错，chunk_size 就是摆设。
# 同一段 114 字文本实测对比：
#   separator=''  -> 3 条，长度 49 / 50 / 23    chunk_size 生效
#   separator=' ' -> 2 条，长度 9 / 102         后一条远超 50，等于没切
#
# 但 '' 只是「能切」，谈不上「切得好」：逐字符切的位置没有任何语义，
# 可能把一句话甚至一个词从中间断开，检索时召回的片段读起来莫名其妙。
# 要按句子边界切，得用 RecursiveCharacterTextSplitter 并自定义分隔符
# （把 。！？ 之类也列进去），或按 token 切的 TokenTextSplitter
#
# 这里切的是裸字符串：输入 str、返回 list[str]。
# 如果手上是 loader 产出的 Document，用 splitter.split_documents(docs)，
# 切完仍是 list[Document]，metadata 会自动复制到每个片段上，方便回溯来源
texts: list[str] = splitter.split_text(text)
rprint(texts)

#  输出：
# [
#     'LangChain
# 是一个用于构建基于大语言模型（LLM）应用的开发框架，旨在帮助开发者更高效地集',
#     '更高效地集成、管理和增强大语言模型的能力，构建端到端的应用程序。它提供了一
# 套模块化工具和接口，支持从',
#     '口，支持从简单的文本生成到复杂的多步骤推理任务'
# ]

for text in texts:
    print(len(text))  # 49 50 23
