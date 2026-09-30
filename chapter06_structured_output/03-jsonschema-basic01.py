import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import SecretStr
from rich import reconfigure, print as rprint

json_shema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "director": {
            "type": "string",
            "description": "The director of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The movie's rating out of 10"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    # 和 Pydantic 那组一样，要指定 tool_choice，所以得关掉思考模式
    extra_body={"thinking": {"type": "disabled"}}
)

# if method == "json_schema":
#     method = "function_calling"

structured_chat = chat.with_structured_output(json_shema, method='json_schema')

# 只给一部电影名，四个字段全让模型自己补
response = structured_chat.invoke('给出盗梦空间的信息')
#
rprint(response)
print(type(response))  # <class 'dict'>，不是对象——取字段得写 response['title']
