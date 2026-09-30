"""用「消息对象」跟模型对话 —— 最原始的调用方式。

前面几个脚本给 invoke() 传的是纯字符串，这里换成**消息列表**：

    SystemMessage   系统消息，设定模型的人设/角色，惯例放第一条
    HumanMessage    用户消息，也就是「我说的话」

比起把内容拼成一个大字符串，用消息列表的好处是角色分明，
模型能区分每句话是谁说的；多轮对话就是把历史消息按顺序累积进这个列表。
后面的 ChatPromptTemplate（见 03-chatPromptTemplate*.py）本质上也是在
帮我们生成这样一个消息列表，只是把「哪些是变量」这件事抽了出来。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import SecretStr
from rich import print as rprint
from utils import save_response

# 读取 .env 文件并写入 os.environ
# override=True：.env 里的值覆盖已存在的同名环境变量（默认不覆盖）
load_dotenv(override=True)

# 取环境变量用下标而不是 .get()：
# .get() 的静态类型是 str | None，而 model 是必填 str；
# 下标取不到会立刻在这一行抛 KeyError，报错点更靠前
MODEL = os.environ["DEEPSEEK_MODEL"]

# SecretStr 不是 str 的子类，运行时由 pydantic 负责转换。
# 显式包一层而不是直接传 str，是为了让「这是密钥」这件事体现在类型上，
# 打印对象时它会显示成 **********，不会把 key 泄到日志里
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')

# 标注成 list[BaseMessage]：列表里会混放不同子类（System / Human，后续还会 append AIMessage），
# 用基类做元素类型，类型检查器就不会因为某个子类不在联合类型里而报警
messages: list[BaseMessage] = [
    # 第一条固定是人设：这里设定成一个「前端转 AI 全栈」的求职者，
    # 后面模型的回答都会站在这个身份上展开
    SystemMessage(content="你是一个求职者，从事前端开发10年，现在要转AI全栈开发"),
    # 两条 HumanMessage 会被当成「用户连续说了两句话」，
    # 模型能看到前一句，所以第二句里的「前端技术栈」指代的是第一句的上下文
    HumanMessage(content="需要准备哪些技术栈"),
    HumanMessage(content="前端技术栈还需要复习吗？")
]

# invoke 入参是消息列表，返回值是 AIMessage
response = chat.invoke(messages)

# 用 rich 的 print 而不是内置 print：AIMessage 的 content 里带换行，
# 内置 print 会把 \n 原样显示成转义字符，rich 会按真实换行渲染，可读性好很多
rprint(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-messages01")
print(f"已写入：{out_path}")
