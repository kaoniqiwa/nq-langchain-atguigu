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
# 效果等价于直接 ChatDeepSeek(...)，好处是换厂商只改这一个字符串。
#
# 除了 model / api_key / model_provider 这三个定位用的参数，其余的都原样透传给底层模型类
# （这里是 ChatDeepSeek），所以写法和直接 new 一个 ChatDeepSeek 完全一样：
#   temperature  采样温度，越大越发散。DeepSeek 官方按场景给过推荐值，代码/数学偏低（接近 0），
#                通用对话约 1.0，翻译、创意写作更高 —— 不确定就先用 1.0
#   max_tokens   单次回复的 token 上限（只算输出，不含输入），默认不限制
#   timeout      单次请求超时秒数，默认不超时 —— 不设的话网络卡住会一直干等
#   max_retries  超时、限流这类可重试错误的自动重试次数，默认 2
chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider="deepseek",
    temperature=1.0,
    max_tokens=1024,
    timeout=30,
    max_retries=2,
)

# invoke 是同步调用，入参可以是纯字符串，也可以是消息列表
response = chat.invoke('介绍下你自己')
# 返回的是 AIMessage，除了 content，还带 usage_metadata、response_metadata 等元信息
print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-params-deepseek01")
print(f"已写入：{out_path}")
