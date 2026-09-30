"""一次抽取多个对象 —— 把「列表结构」交给 with_structured_output。

前面的文件抽的都是单个对象（Person）。实际场景更常见的是「一段话里有好几个人」，
这时要让模型返回一个**列表**。做法很简单：用一个外层模型把列表包起来——

    class PersonList(BaseModel):
        persons: list[Person]

把它传给 with_structured_output，拿到的就是 PersonList 实例，
里面的 persons 是一个个 Person，不用自己拆字符串、也不用循环解析。

和 01-pydantic-basic02.py 一样，这里也关掉了思考模式：
function_calling 要指定 tool_choice，而 DeepSeek 的思考模式不支持。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import SecretStr, BaseModel
from rich import reconfigure, print as rprint

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


class Person(BaseModel):
    """人物信息"""
    name: str
    age: int


class PersonList(BaseModel):
    """人物列表信息"""
    # 嵌套结构：外层模型的字段类型是「另一个模型的列表」
    persons: list[Person]


# 注意区分类方法和实例方法，这是 pydantic 上很容易踩的坑：
#   Person.model_json_schema()   类方法，直接对类调用，返回 JSON Schema
#   Person.model_dump()          实例方法，必须先造出实例再调
# 原来写成 Person.model_dump() 会报
#   missing 1 required positional argument: 'self' —— 没有实例，self 没地方来
p = Person(name='张三', age=30)
print(p.model_dump())  # {'name': '张三', 'age': 30}

# 类方法可以直接对类调用：Person 是平的，打印出来就是 properties + required
print(Person.model_json_schema())

# 换成外层模型才看得到嵌套是怎么表达的：list[Person] 被写成
# {"items": {"$ref": "#/$defs/Person"}, "type": "array"}，
# 而 Person 的定义被挪到 $defs 里单独存一份，用 $ref 指过去
print(PersonList.model_json_schema())

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    # 关掉思考模式的原因同 01-pydantic-basic01.py：要指定 tool_choice
    extra_body={"thinking": {"type": "disabled"}}
)

# 传进去的是**外层**模型 PersonList。
# 也就是说被强制调用的那个「工具」名字是 PersonList，里层的 Person
# 只是它的一个字段类型，不会单独成为一个工具
structured_chat = chat.with_structured_output(PersonList)

# 一句话里有两个人，让模型分别抽出来装进 persons 列表
response = structured_chat.invoke('张三30岁,李四 25岁')

rprint(response)
print(type(response))  # <class '__main__.PersonList'>
print(response.persons)  # [Person(name='张三', age=30), Person(name='李四', age=25)]
