import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, AutoStrategy
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
agent = create_agent(chat, name='chat-assistant', response_format=AutoStrategy(Person))

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

out_path = save_response(response, "02-agent-advanced04")
print(f"已写入：{out_path}")
