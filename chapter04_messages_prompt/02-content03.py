"""读取「思考模式」下模型返回的内容 —— 同一个响应的三种看法。

开启 thinking 后，一次响应里其实有两部分：

    思维链（reasoning）   模型自己推理的过程
    最终回答             面向用户的那段话

AIMessage 上取这两部分有三条路，本脚本依次打印对比：

    additional_kwargs['reasoning_content']   供应商原始字段，deepseek 特有
    content                                  最终回答，最常用
    content_blocks                           langchain 归一化后的块列表，
                                             思维链被标成 reasoning 类型的块

跨模型时用 content_blocks 更稳，因为它是 langchain 统一过的结构，
不必去记各家供应商的字段名。

注意：思考模式下返回的 reasoning_content，如果要把这条 AIMessage 再发回去，
必须原样带回，否则 API 会报 400。本脚本只问一轮就结束，不涉及回传。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from pydantic import SecretStr

from utils import save_response

load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

chat = init_chat_model(
    model=MODEL,
    api_key=API_KEY,
    model_provider='deepseek',
    extra_body={"thinking": {"type": "enabled"}}  # 开启思考模式
)

messages: list[BaseMessage] = [
    HumanMessage("你好")
]

# 标注成 AIMessage：invoke 的静态返回类型是 BaseMessage，
# 补上具体类型后面访问 content_blocks 之类才不会被类型检查器拦
response: AIMessage = chat.invoke(messages)

# 第 1 种：供应商原始字段。deepseek 把思维链放在 additional_kwargs 里，
# 这是它自己的约定，换一家模型就没有这个键了（.get 取不到时返回 None，不会报错）
print('=' * 20, '-> reasoning_content <-', '=' * 20)
print(response.additional_kwargs.get('reasoning_content'))

# 第 2 种：content，只有最终回答，不含思维链——平时最常用的就是这个
print('=' * 20, '-> response.content <-', '=' * 20)
print(response.content)

# 第 3 种：content_blocks，langchain 归一化后的块列表
# content_blocks 可以获取模型的思维链
print('=' * 20, '-> response.content_blocks <-', '=' * 20)
print(response.content_blocks)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "02-content03")
print(f"已写入：{out_path}")
