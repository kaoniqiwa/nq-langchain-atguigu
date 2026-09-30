"""ToolStrategy 的第一种 schema —— Pydantic 模型。

ToolStrategy 用「让模型调用一个人工工具」的方式产出结构化数据。
它一共只有三个可调参数（langchain/agents/structured_output.py:196）：

    schema                要什么结构。Pydantic 模型 / dataclass / TypedDict /
                          Union 类型 / 原始 JSON schema dict 都行
    tool_message_content  这个人工工具「被调用」后，回给模型的 ToolMessage 里写什么
    handle_errors         校验不过怎么办，默认 True —— 把错误回给模型让它重试

schema 传 dict 和传 Pydantic 有个关键差别，源码里的 warning 专门写了：
**dict 不做校验**，模型填的参数原样给你，`handle_errors` 也跟着失效。
想要校验、想要自动重试，就得用 Pydantic 模型、dataclass 或 TypedDict。

本文件是最基础的一版：schema 传一个 Pydantic 模型。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import SecretStr, BaseModel, Field
from rich import reconfigure, print as rprint

from utils import save_response

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


# schema 传 Pydantic 模型（这是三种写法里校验最完整的一种）：
# 类名 → 工具名，类 docstring → 工具描述，字段 → 参数 schema
class Person(BaseModel):
    """人物信息"""
    # 字段说明会写进 schema 交给模型，写清楚能明显提升抽取准确率
    name: str = Field(description='姓名')
    age: int = Field(description='年龄')
    occupation: str = Field(description='职业')


# ToolStrategy 会强制 tool_choice，而 DeepSeek 的思考模式不接受，
# 所以模型必须自己构造（字符串形式带不了 extra_body）
chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    # 关掉思考模式绕开 tool_choice 的限制——结构化抽取不需要思维链
    extra_body={"thinking": {"type": "disabled"}}
)

# name 是这张图的名字（CompiledStateGraph 的 name），多 Agent 场景里用来标识它
agent = create_agent(chat, name='chat-assistant', response_format=ToolStrategy(schema=Person))

response = agent.invoke({
    "messages": [{"role": "user", "content": "张三是一名30岁的软件工程师"}]
})

rprint(response)

# 结构化结果单独放在这个键里，不在 messages 中
rprint(response['structured_response'])
print(type(response['structured_response']))  # 看看是 Person 实例还是 dict

# 逐条打印，看清「用户 → 工具调用 → 工具结果 → 最终回答」这条链
for message in response['messages']:
    message.pretty_print()

out_path = save_response(response, "03-agent-toolstrategy01")
print(f"已写入：{out_path}")
