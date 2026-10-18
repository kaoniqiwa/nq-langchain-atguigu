import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

from utils import save_response

# override=True：.env 里的值覆盖已存在的同名环境变量（默认不覆盖）
load_dotenv(override=True)

# 取环境变量用下标而不是 .get()：
# .get() 的静态类型是 str | None，而 model 是必填 str；
# 下标取不到会立刻在这一行抛 KeyError，报错点更靠前，类型检查器也不用再纠结 None
MODEL = os.environ["OPENROUTER_MODEL"]

# openrouter_api_key 字段声明为 SecretStr | None，而 SecretStr 并不是 str 的子类，
# 所以显式包一层，让运行时和类型检查器看到的是同一个类型。
# 其实 pydantic 在校验时会自动把 str 转成 SecretStr，这层包装主要是给类型检查器看的。
# 好处是 SecretStr 的 repr 固定为 **********，key 不会被日志或报错信息打印出来
API_KEY = SecretStr(os.environ["OPENROUTER_API_KEY"])

# 调用线上模型最核心的三个参数：model（模型名）、api_key（鉴权）、base_url（接口地址）
# OpenRouter 是聚合网关，model 传「厂商/模型」形式的完整标识，base_url 不传时走 SDK 默认端点
llm_openrouter = ChatOpenRouter(model=MODEL, api_key=API_KEY)

# invoke 是同步调用，入参可以是纯字符串，也可以是消息列表
response = llm_openrouter.invoke('一句话介绍下你自己')
# 返回的是 AIMessage，除了 content，还带 usage_metadata、response_metadata 等元信息
print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-model-openrouter01")
print(f"已写入：{out_path}")
