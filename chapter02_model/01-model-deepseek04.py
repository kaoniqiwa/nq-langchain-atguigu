import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig

from utils import save_response

# 读取 .env 文件并写入 os.environ
# override=True：.env 里的值覆盖已存在的同名环境变量（默认不覆盖）
load_dotenv(override=True)

# 取环境变量用下标而不是 .get()：
# .get() 的静态类型是 str | None，而 model 是必填 str；
# 下标取不到会立刻在这一行抛 KeyError，报错点更靠前
MODEL = os.environ["DEEPSEEK_MODEL"]


# 这里演示「动态配置」写法：构造时不绑模型，模型名留到 invoke 时通过 config 传，
# 一个实例就能服务多个模型。
#
# 括号里什么都不传 —— init_chat_model 此时默认只开放 model / model_provider
# 两个可配置字段，provider 会从模型名前缀推断（deepseek-flash → deepseek）。
#
# api_key 不在这里传：ChatDeepSeek 的字段自带 default_factory，
# 会自己从 DEEPSEEK_API_KEY 环境变量读（该变量来自 shell，不在 .env 里）。
# 硬塞进 config 也不生效——它不在上面那两个可配置字段之内。
chat = init_chat_model()

# invoke 是同步调用，入参可以是纯字符串，也可以是消息列表
# config 里的 model 在运行时生效，覆盖构造时的设置 —— 这正是上面不绑模型的前提，
# 同一个实例换个 config 就能切到别的模型，不必为每个模型各建一个对象
response = chat.invoke('一句话介绍下你自己',
                       config=RunnableConfig(configurable={"model": MODEL}))
# 返回的是 AIMessage，除了 content，还带 usage_metadata、response_metadata 等元信息
print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-model-deepseek04")
print(f"已写入：{out_path}")
