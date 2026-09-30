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
# 下标取不到会立刻在这一行抛 KeyError，报错点更靠前，类型检查器也不用再纠结 None
MODEL = os.environ["OPENROUTER_MODEL"]

# api_key 是 ChatOpenRouter 的字段别名（真名叫 openrouter_api_key），
# 类型是 SecretStr | None，而 SecretStr 并不是 str 的子类，
# 所以显式包一层，让运行时和类型检查器看到的是同一个类型。
# 其实 pydantic 在校验时会自动把 str 转成 SecretStr，这层包装主要是给类型检查器看的。
# 好处是 SecretStr 的 repr 固定为 **********，key 不会被日志或报错信息打印出来
API_KEY = SecretStr(os.environ["OPENROUTER_API_KEY"])

# 调用线上模型最核心的三个参数：model（模型名）、api_key（鉴权）、base_url（接口地址）
#
# model_provider 这里必须显式写 'openrouter'。
# 因为 init_chat_model 会按模型名的前缀猜 provider，而 .env 里
# OPENROUTER_MODEL 的值是 gemini-3.8-flash —— gemini 开头会被猜成
# google_vertexai，于是去加载 langchain-google-vertexai，报
#   ImportError: Initializing ChatVertexAI requires the langchain-google-vertexai package
# 写成 'openrouter:gemini-3.8-flash' 这种前缀形式也可以，效果一样
#
# base_url 不用传：ChatOpenRouter 会自己走 OpenRouter 的官方端点。
# 为什么"不传也能跑"，见同目录的 base-url-defaults.md
llm_openrouter = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='openrouter',
    temperature=0.6,
)

# invoke 是同步调用，入参可以是纯字符串，也可以是消息列表
response = llm_openrouter.invoke('一句话介绍下你自己')
# 返回的是 AIMessage，除了 content，还带 usage_metadata、response_metadata 等元信息
print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-model-openrouter03")
print(f"已写入：{out_path}")
