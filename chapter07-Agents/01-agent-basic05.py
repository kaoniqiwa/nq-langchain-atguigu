import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import SecretStr
from rich import reconfigure, print as rprint
from utils import save_response

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

tool_search = TavilySearch(max_results=4, tavily_api_key=TAVILY_API_KEY)


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取指定城市的天气。

    Args:
        city: 具体的城市
    """
    return f"{city} 的天气是晴朗，万里无云"


agent = create_agent(
    model=MODEL,
    tools=[get_weather, tool_search],
    system_prompt="你是一个 helpful 助手",
)

response = agent.invoke({
    "messages": [
        HumanMessage('今天北京的天气？'),
        HumanMessage('今日油价？')
    ]
})
rprint(response)

out_path = save_response(response, "01-agent-basic05")
print(f"已写入：{out_path}")
