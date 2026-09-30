import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import SecretStr
from rich import reconfigure, print as rprint
from utils import save_response

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

flag = 0


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取指定城市的天气。

    Args:
        city: 具体的城市
    """
    global flag
    flag += 1
    if flag < 3:
        return 'TEMP_UNAVAILABLE: 天气服务暂时不可用，请稍后重试'
    return f"{city} 的天气是晴朗，万里无云"


agent = create_agent(
    model=MODEL,
    tools=[get_weather],
    system_prompt="""
你是一个天气助手。
当工具返回以 'TEMP_UNAVAILABLE:' 开头的结果时， 说明是临时故障，不要立即放弃; 你应再次调用同一个工具，最多重试 3 次。
如果 3 次后仍失败，再向用户说明服务暂时不可用。
""",
)

response = agent.invoke({
    "messages": [
        HumanMessage('今天北京的天气？'),
    ]
})
rprint(response)

out_path = save_response(response, "01-agent-basic06")
print(f"已写入：{out_path}")
