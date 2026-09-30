import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from pydantic import SecretStr
from rich import reconfigure, print as rprint
from utils import save_response

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

agent = create_agent(MODEL, name='chat-assistant')
response = agent.invoke({
    "messages": [{"role": "user", "content": "你好，1+2=?"}]
})

for message in response['messages']:
    message.pretty_print()

out_path = save_response(response, "01-agent-advanced01")
print(f"已写入：{out_path}")
