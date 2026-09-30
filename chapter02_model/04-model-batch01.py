import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from utils import save_response


# ============================================================
# 本脚本演示「批量调用」
#
# batch() 一次提交多条输入，内部并发发起请求，最后把结果按顺序收成一个列表。
# 适合「有一堆互不依赖的问题要问」的场景：比写 for 循环逐个 invoke 快得多，
# 因为多个请求是同时在跑的，总耗时约等于最慢的那一条，而不是各自耗时之和。
#
# 返回值和入参一一对应：第 N 条输入的结果就是返回列表的第 N 项。
# ============================================================

# 读取 .env 文件并写入 os.environ
# override=True：.env 里的值覆盖已存在的同名环境变量（默认不覆盖）
load_dotenv(override=True)

# 取环境变量用下标而不是 .get()：
# .get() 的静态类型是 str | None，而 model 是必填 str；
# 下标取不到会立刻在这一行抛 KeyError，报错点更靠前，类型检查器也不用再纠结 None
MODEL = os.environ["DEEPSEEK_MODEL"]

# api_key 字段声明为 SecretStr | None，而 SecretStr 并不是 str 的子类，
# 所以显式包一层，让运行时和类型检查器看到的是同一个类型。
# 其实 pydantic 在校验时会自动把 str 转成 SecretStr，这层包装主要是给类型检查器看的。
# 好处是 SecretStr 的 repr 固定为 **********，key 不会被日志或报错信息打印出来
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

# 调用线上模型最核心的三个参数：model（模型名）、api_key（鉴权）、base_url（接口地址）
# base_url 不传时默认走 https://api.deepseek.com
llm_deepseek = ChatDeepSeek(model=MODEL, api_key=API_KEY)


# batch() 的入参是一个可迭代对象，每一项的写法和 invoke 的入参一样：
# 可以是纯字符串，也可以是消息列表。返回的是 list[AIMessage]，长度和入参相同。
response = llm_deepseek.batch([
    "你好，你是谁",
    "中国的首都是哪里",
    "2+3 = ？"
])

# response 是列表，直接 print 会打印整个对象的 repr（换行被转义成 \n）
# 只看正文用逐条取 content：
for msg in response:
    print(msg.content)

# 把三条响应合并落盘成一个 JSON 数组，实现见 utils.save_response
# save_response 两种入参都收：单条消息存成一个 JSON 对象，列表存成一个 JSON 数组，
# 数组元素的结构和单条时完全一致，所以合并存不影响后续读取
out_path = save_response(response, "04-model-batch01")
print(f"已写入：{out_path}")


