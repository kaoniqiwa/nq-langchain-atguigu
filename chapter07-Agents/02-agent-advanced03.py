"""Agent 的结构化输出 —— response_format 的三种策略。

Agent 默认输出一段自然语言，放在 response["messages"][-1] 里。
想让它同时给出一份结构化数据，就用 create_agent 的 response_format 参数。
三种策略（langchain/agents/structured_output.py）：

    ToolStrategy(Person)      把 Person 做成一个「人工工具」，
                              模型通过调用它来输出结构化数据 —— 本质是 tool calling
    ProviderStrategy(Person)  用厂商原生的结构化输出接口（OpenAI 的 json_schema 那套）
    AutoStrategy(Person)      让框架自己挑

本脚本用的是 ToolStrategy。它和第六章的 with_structured_output 是同一件事的
两个入口，method 参数正好一一对应：

    method="function_calling"  ≈  ToolStrategy
    method="json_schema"       ≈  ProviderStrategy

区别只在于约束的对象：第六章约束「模型的输出」，这里约束「Agent 的输出」。

结果放在哪 —— 不在 messages 里，而是单独的键：

    response["messages"]             完整对话历史（含工具调用、工具结果）
    response["structured_response"]  解析好的结构化结果

注意 ToolStrategy 会让 create_agent 用 tool_choice="any" 绑定工具
（factory.py:1422，注释写着 Force tool use），也就是强制模型必须调某个工具。
DeepSeek 的思考模式不接受 tool_choice —— 和第六章那个 400 同源，实测会报：

    400 - Thinking mode does not support this tool_choice

所以下面不能直接传模型名字符串（那样内部走 init_chat_model(model)，
带不了 extra_body），得先自己构造模型、关掉思考模式，再交给 create_agent。
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


class Person(BaseModel):
    """人物信息"""
    # 字段说明会写进 schema 交给模型，写清楚能明显提升抽取准确率
    name: str = Field(description='姓名')
    age: int = Field(description='年龄')
    occupation: str = Field(description='职业')


# 不能直接传模型名字符串：那样内部走 init_chat_model(model)，带不了 extra_body，
# 而 ToolStrategy 又强制 tool_choice，必然 400。所以先自己把模型构造好
chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    # 关掉思考模式绕开 tool_choice 的限制——结构化抽取不需要思维链
    extra_body={"thinking": {"type": "disabled"}}
)

# name 是这张图的名字（CompiledStateGraph 的 name），多 Agent 场景里用来标识它
agent = create_agent(chat, name='chat-assistant', response_format=ToolStrategy(Person))

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

out_path = save_response(response, "02-agent-advanced03")
print(f"已写入：{out_path}")
