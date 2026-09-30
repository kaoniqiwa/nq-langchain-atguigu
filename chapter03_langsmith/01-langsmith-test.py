import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr


# 读取 .env 文件并写入 os.environ
# override=True：.env 里的值覆盖已存在的同名环境变量（默认不覆盖）
load_dotenv(override=True)

# 取环境变量用下标而不是 .get()：
# .get() 的静态类型是 str | None，而 model 是必填 str；
# 下标取不到会立刻在这一行抛 KeyError，报错点更靠前
MODEL = os.environ["DEEPSEEK_MODEL"]

# api_key 字段声明为 SecretStr | None，而 SecretStr 并不是 str 的子类，所以显式包一层。
# 这里 key 经 **kwargs 透传给 ChatDeepSeek，pydantic 校验时会自动把 str 转成 SecretStr，
# 包不包都能跑；与 01 保持一致，顺带得到 SecretStr 的 repr 固定为 ********** 的好处
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


# init_chat_model 是统一入口，实例化哪家的 ChatModel 由 model_provider 决定，
# 效果等价于 01 里直接 ChatDeepSeek(model=..., api_key=...)，好处是换厂商只改这个字符串。
#
# 这里演示的是「动态配置」写法：构造时不绑模型，模型名留到 invoke 时通过 config 传，
# 一个实例就能服务多个模型。官网对应写法：configurable_model = init_chat_model(temperature=0)
# 括号里可以什么都不传 —— 模型名由 config 提供，api_key 由 init_chat_model 从
# DEEPSEEK_API_KEY 环境变量读取（load_dotenv 已经把它放进 os.environ）
chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')

# invoke 是同步调用，入参可以是纯字符串，也可以是消息列表
# config 里的 model 在运行时生效，覆盖构造时的设置 —— 这正是上面不绑模型的前提，
# 同一个实例换个 config 就能切到别的模型，不必为每个模型各建一个对象
response = chat.invoke('一句话介绍下你自己', config=RunnableConfig(run_name="joke_generation",tags=["my_tag1"]))
print(response)
