import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import SecretStr

from utils import save_response

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
# 下面两行注释掉的是省略 model_provider 的写法，留作对照 —— 本机（langchain 1.4.0）实测
# 两种都能推断出 deepseek，构造出来的同样是 ChatDeepSeek：
#   只给模型名 —— 靠前缀推断，base.py 里 startswith("deepseek") 归到 deepseek
#   "厂商:模型名" —— 由 _parse_model 按第一个冒号拆成模型名 + 厂商
# 推断是 best-effort，换模型名就可能认不出来；requirements.txt 锁的 1.2.12 也未必认这两条
# chat = init_chat_model(model="deepseek-flash", api_key=API_KEY)
# chat = init_chat_model(model="deepseek:deepseek-flash", api_key=API_KEY)
# 这里仍显式传 model_provider：不依赖前缀推断，模型名怎么起都不影响
chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')

# invoke 是同步调用，入参可以是纯字符串，也可以是消息列表
response = chat.invoke('一句话介绍下你自己')
# 返回的是 AIMessage，除了 content，还带 usage_metadata、response_metadata 等元信息
print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-model-deepseek03")
print(f"已写入：{out_path}")
