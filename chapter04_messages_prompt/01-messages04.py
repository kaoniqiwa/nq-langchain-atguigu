"""工具调用（function calling）的消息流 —— 手工构造一段工具调用历史。

真实场景里 tool_calls 是模型自己生成的，这里**手写**出来，只为看清一次完整的
工具调用在「消息层面」长什么样、顺序怎么排：

    1. HumanMessage                                   用户提问
    2. AIMessage(content="", tool_calls=[ToolCall])   模型决定调哪个工具、传什么参数
                                                      注意此时还没有回答用户，content 是空的
    3. ToolMessage(content=..., tool_call_id=...)     我们执行完工具，把结果回填
    4. 再 invoke() 一次，模型拿着工具结果生成最终回答

两个必须对齐的字段：ToolCall 的 id 和 ToolMessage 的 tool_call_id 是同一个值
（这里都是 call_123）。API 靠它把「工具执行结果」和「哪一次调用」配对，
对不上会直接报错。

本脚本没有真的去查天气，ToolMessage 里的内容是写死的。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolCall, ToolMessage
from pydantic import SecretStr

load_dotenv(override=True)

MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='deepseek',
    # 关掉思考模式：开启时模型会额外返回 reasoning_content，
    # 而这个字段在下一轮回传时必须原样带上，否则报 400。
    # 工具调用这条链路不需要思维链，直接关掉最省事
    extra_body={"thinking": {"type": "disabled"}}
)

# 手动拼出「已经走完一轮工具调用」的状态，再 invoke 一次拿最终回答
messages: list[BaseMessage] = [
    # 第 1 条：用户提问
    HumanMessage('今天北京的天气怎样'),
    # 第 2 条：模型的决策——content 为空，因为在回答用户之前它先要调工具。
    # args 是传给 get_weather 的参数，id 是这次调用的唯一标识
    AIMessage(content="", tool_calls=[
        ToolCall(type="tool_call", name='get_weather', args={"location": "中国北京"}, id="call_123")]),
    # 第 3 条：工具的执行结果。tool_call_id 必须等于上面那个 id，
    # 内容就是工具真实会返回的东西（这里是假数据）
    ToolMessage(content="晴朗，万里无云", tool_call_id="call_123")
]

# 带着上面这段历史提问：模型看到工具已经返回了结果，
# 这一轮就会停掉 tool_calls，直接用 content 给出面向用户的回答
response = chat.invoke(messages)
print(response)
