import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.chat_models import init_chat_model
from pydantic import SecretStr, BaseModel, Field
from rich import reconfigure, print as rprint
from utils import save_response

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["OPENROUTER_MODEL"]

API_KEY = SecretStr(os.environ["OPENROUTER_API_KEY"])


class Person(BaseModel):
    """人物信息"""
    # Field 里的 description 会写进 schema 交给模型，
    # 相当于告诉模型「这个字段该填什么」，写清楚能明显提升抽取准确率
    name: str = Field(description='姓名')
    age: int = Field(description='年龄')
    occupation: str = Field(description='职业')


llm_openrouter = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='openrouter',
    temperature=0.6,
)
# DeepSeek 不支持原生结构化输出
agent = create_agent(llm_openrouter, name='chat-assistant', response_format=ProviderStrategy(Person))
response = agent.invoke({
    "messages": [{"role": "user", "content": "张三是一名30岁的软件工程师"}]
})
#
rprint(response)
for message in response['messages']:
    message.pretty_print()

out_path = save_response(response, "01-agent-advanced01")
print(f"已写入：{out_path}")
