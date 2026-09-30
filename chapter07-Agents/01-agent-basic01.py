"""最小的 Agent —— 只有一个模型，没有工具。

create_agent 是 LangChain 1.0 里创建 Agent 的入口，签名大致是：

    create_agent(model, tools=None, *, system_prompt=None, middleware=(),
                 response_format=None, checkpointer=None, ...)

它返回的不是模型，而是一张**编译好的 LangGraph 图**（CompiledStateGraph）。
因为 Agent 的执行过程本身就是个循环：

    模型决定下一步做什么 → 执行 → 把结果喂回去 → 再决策 → ……
    直到模型认为可以给出最终回答

这种「带状态的循环」用图来描述最自然，所以 Agent 底层就是 LangGraph。
类型是 CompiledStateGraph 而不是 BaseChatModel，原因就在这。

本脚本只做一件事：把 Agent 建出来，看看它是什么类型。
还没给它任何工具，所以它现在只能靠模型自身的知识回答，
和直接调模型差别不大——加工具见后面的文件。
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from pydantic import SecretStr
from rich import reconfigure, print as rprint
from utils import save_response

reconfigure(force_terminal=True)
load_dotenv(override=True)
MODEL = os.environ["DEEPSEEK_MODEL"]

# 这行现在没用到：下面直接把模型名字符串交给了 create_agent，
# 里面的 init_chat_model 会按 ChatDeepSeek 的 default_factory 去读
# DEEPSEEK_API_KEY 环境变量，不需要手动传 key。
# 一旦改成「自己构造模型再传进去」（见下面注释里的第 1 种写法），这里就必须要用了
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])

# 第一个参数是模型：给字符串的话，create_agent 内部会走
#     model = init_chat_model(model)        # langchain/agents/factory.py:996
# 和直接调 init_chat_model 完全等价，包括那套「按名字前缀推断 provider」的逻辑。
#
# 坑在这里：模型名不是 deepseek-* 这类能推对的，就会推错。
# 但 create_agent 的签名里**没有 model_provider 参数**，所以只有两条路：
#
#   1. 自己构造好再传：
#        model = init_chat_model(model=MODEL, api_key=API_KEY,
#                                model_provider="openrouter")
#        agent = create_agent(model)
#   2. 用 provider:model 的前缀形式（init_chat_model 支持）：
#        agent = create_agent("openrouter:gemini-3.8-flash")
#
# 具体踩坑经过见 chapter02_model/01-model-openrouter03.py
#
# 第二个参数是 tools，省略等于一个没有任何工具的 Agent
agent = create_agent(MODEL)
response = agent.invoke({
    "messages": [{"role": "user", "content": "你好，1+2=?"}]
})

rprint(type(agent))  # <class 'langgraph.graph.state.CompiledStateGraph'>
rprint(response)

# 落盘。注意 response 不是一条消息，而是 Agent 的状态字典：
#     {"messages": [HumanMessage, AIMessage, ...]}
# save_response 会递归处理它——键保留，列表里的每条消息各自转成 dict。
# 这也是 Agent 和普通模型调用的一个区别：一次 invoke 得到的是一整段消息历史，
# 中间可能有工具调用、工具结果，全部都在 messages 里
out_path = save_response(response, "01-agent-basic01")
print(f"已写入：{out_path}")
