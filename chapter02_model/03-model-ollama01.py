import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from utils import save_response


# 读取 .env 文件并写入 os.environ
# override=True：.env 里的值覆盖已存在的同名环境变量（默认不覆盖）
load_dotenv(override=True)

# 取环境变量用下标而不是 .get()：
# .get() 的静态类型是 str | None，而 model 是必填 str；
# 下标取不到会立刻在这一行抛 KeyError，报错点更靠前，类型检查器也不用再纠结 None
MODEL = os.environ["OLLAMA_MODEL"]

# 调用线上模型最核心的三个参数：model（模型名）、api_key（鉴权）、base_url（接口地址）
# Ollama 跑在本机，不需要 api_key；base_url 也不用传——
# ChatOllama 的 base_url 默认是 None，透传给 ollama SDK 后会依次尝试
# OLLAMA_HOST 环境变量 → 默认 http://127.0.0.1:11434。
# 只有 Ollama 跑在别的机器或非默认端口时，才需要显式传 base_url
llm_ollama = ChatOllama(model=MODEL)


# invoke 是同步调用，入参可以是纯字符串，也可以是消息列表
response = llm_ollama.invoke('介绍下你自己')
# 返回的是 AIMessage，除了 content，还带 usage_metadata、response_metadata 等元信息
print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "03-model-ollama01")
print(f"已写入：{out_path}")
