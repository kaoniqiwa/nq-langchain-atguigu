import os

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from utils import save_response


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


# ============================================================
# 本脚本演示「多轮会话的历史记录」机制
#
# 模型本身不保存任何状态：每次 invoke 都是一次独立的无状态请求，
# 服务端不会记得你上一句问了什么。所谓「模型记得上文」，
# 完全是因为我们每次都把完整的历史消息列表重新传了过去。
#
# 所以多轮对话 = 自己维护一个 messages 列表：
# 每轮把用户输入 append 进去，再把模型的回答 append 进去，下轮一起发。
# 少 append 任何一条，模型就「失忆」。
#
# 注解写成 list[BaseMessage] 而不是让它自动推断：
# 列表后续要 append 进 AIMessage / HumanMessage 等不同类型，
# 显式标注成基类列表，类型检查器才不会在 append 时报「类型不匹配」。
# ============================================================

# 第一轮：只有 system 设定角色 + 用户提问
messages: list[BaseMessage] = [
    SystemMessage('你是一个专业的数学老师'),
    HumanMessage('1+2=?'),
]
response = llm_deepseek.invoke(messages)
# 返回的是 AIMessage，除了 content，还带 usage_metadata、response_metadata 等元信息
print(response)

# 第二轮：把上一轮的问和答都追加进历史，再提新问题
# response 本身就是 AIMessage，直接 append 即可，不要写成 AIMessage(response.content) 去重建——
# 那样会丢掉 id、usage_metadata 等字段，尤其是 tool_calls：
# 做工具调用时 history 里缺了 tool_calls，下一轮请求会因为「工具调用没有对应结果」直接报错
messages.append(response)
messages.append(HumanMessage('我刚才问了什么问题'))

# 这次请求带上了完整历史，模型才能答出「你刚才问的是 1+2」
# 如果把上面两行 append 注释掉再跑，模型就会答不上来——这就是历史记录的作用
response = llm_deepseek.invoke(messages)
print(response)


# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
# 注意：落盘的是第二轮的 response，第一轮的没有保存
out_path = save_response(response, "04-model-invoke03")
print(f"已写入：{out_path}")
