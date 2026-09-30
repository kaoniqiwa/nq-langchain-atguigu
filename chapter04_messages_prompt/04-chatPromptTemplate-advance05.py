"""用 MessagesPlaceholder 存历史消息，实现多轮对话。

这个脚本把前面几个知识点串起来了：

    ChatPromptTemplate   定义模板骨架
    MessagesPlaceholder  在骨架里给「历史消息」留一个位置
    invoke()             填充变量，得到完整的消息列表

模板骨架是：system（固定人设）+ history（历史，可变长）+ human（本次提问）。
每次对话只需把累积的历史塞进 history，模板本身不用改——这就是它比手写
messages 方便的地方。

注意这里用的是显式构造的 MessagesPlaceholder('history')，它默认 optional=False，
也就是**必填**：invoke 时不给 history 会报错。
对照 04-chatPromptTemplate-advance02.py 里的简写 ("placeholder", "{x}")——
那种写法是**可选**的，不传就是空列表。两者行为不同，按需要选。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import SecretStr
from rich.console import Console

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "我是一个AI助手"),
    # 历史消息的占位符：填进去的是一个长度可变的**消息列表**，不是一段文本。
    # 显式构造的 MessagesPlaceholder 默认必填，所以下面 invoke 时必须给 history
    MessagesPlaceholder('history'),
    ("human", "{question}")
])
print('chat_prompt_template:\n', chat_prompt_template)

# history 用元组形式给，会被转成真正的 HumanMessage / AIMessage，角色完整保留；
# question 是本次提问——模型要结合上面的「5+2=7」才能算出 7*4=28
prompt_value = chat_prompt_template.invoke(
    {
        "history": [
            ("human", "5+2=?"),
            ("ai", "5+2=7")
        ],
        "question": "结果再乘以4呢"
    }
)

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='deepseek',
    extra_body={"thinking": {"type": "enabled"}}  # 开启思考模式
)

# prompt_value 是 PromptValue，内含 system / human / ai 多条消息，
# 角色信息完整保留，直接交给 chat.invoke() 即可
response = chat.invoke(prompt_value)
Console(force_terminal=True).print(response)
