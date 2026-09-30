import os
from collections.abc import Iterator

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from utils import save_response

# ============================================================
# 本脚本演示「流式输出」
#
# invoke() 是一次性拿到完整回答，中间白屏干等；
# stream() 则是服务端边生成边推送，本地收到一小片就能立刻打印，
# 这就是 ChatGPT 那种「打字机效果」的实现方式。
#
# 代价是拿到的东西不再是一条完整消息，而是一串分片：
# 每个分片是一个 AIMessageChunk，需要自己循环消费。
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

# 关键差别：stream() 返回的是迭代器，不是 AIMessage
# 它一个 chunk 一个 chunk 地吐，chunk.text 是这一片新生成的文本片段
response: Iterator[AIMessageChunk] = llm_deepseek.stream('什么是人工智能')

# 循环消费分片，边收边打印
# full：把分片重新拼成一条完整消息。AIMessageChunk 支持 + 运算，
#       拼接后 content 是完整正文，usage_metadata 等字段也会一并合并，
#       效果等价于一次 invoke 的返回值，这样才能落盘
# end=''      ：取消 print 默认的换行，否则每个分片都会另起一行，刷成一列竖排
# flush=True  ：强制立刻刷新输出缓冲区，否则内容会攒在缓冲里成批显示，就失去流式的意义了
full = None
for chunk in response:
    full = chunk if full is None else full + chunk
    print(chunk.text, end='', flush=True)
# 兜底：一个分片都没收到说明请求异常，早点报错，别让后面的落盘拿到 None
if full is None:
    raise ValueError('no response from DeepSeek')

# 把拼接好的完整消息落盘成 JSON，实现见 utils.save_response
# 注意传的是 full 而不是 response：response 是迭代器，上面已经把它耗尽了，
# 而且迭代器不是 AIMessage，save_response 要的是消息对象
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(full, "04-model-stream01")
print(f"已写入：{out_path}")
