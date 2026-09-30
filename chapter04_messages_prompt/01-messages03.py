"""多轮对话 + 流式输出 + 历史落盘。

一个可以一直聊下去的命令行小程序，把前面几个知识点串起来：

    while 循环             持续接收输入，每轮把用户消息 append 进 messages
    keep_recent_messages   发送前裁剪，只发 system + 最近 MAX_PAIRS_HISTORY 轮
    chat.stream()          流式输出，边生成边打印，不用等整段回复生成完
    save_history()         每轮把完整历史覆盖写入 output/01-messages03-history.json

三种退出方式：输入 quit / exit，或按 Ctrl+D（macOS / Linux；
Windows 是 Ctrl+Z 再回车）。落盘是覆盖式的——文件名固定不带时间戳，
所以文件里始终只有最新的一份历史快照。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessageChunk
from pydantic import SecretStr

from utils import keep_recent_messages, save_history

load_dotenv(override=True)

# 退出词单独提成常量：要加 "q"、"bye" 只改这一处，下面判断和提示语自动跟着变
EXIT_WORDS = ("quit", "exit")
# 发给模型时最多保留几轮对话。注意只影响「发给模型的那份」，
# 落盘的 messages 仍是完整历史，不受这个上限影响
MAX_PAIRS_HISTORY = 10
MODEL = os.environ["DEEPSEEK_MODEL"]
API_KEY = SecretStr(os.environ["DEEPSEEK_API_KEY"])


chat = init_chat_model(model=MODEL, api_key=API_KEY, model_provider='deepseek')

# 开局只有一条 system：人设在整个会话里置顶保留，裁剪时也不会被切掉
messages: list[BaseMessage] = [
    SystemMessage(content="你是小谷姐姐，是尚硅谷教育的数字员工，也是一名耐心、友好的AI助手，可以回答学员的问题"),
]
print(f"请输入具体的问题，输入 {' 或 '.join(EXIT_WORDS)}（或按 Ctrl+D）结束对话")

i = 1
while True:
    print("\n", "=" * 10, f"第{i}轮对话开始", "=" * 10, "\n")
    try:
        user_input = input('请输入：')
    except EOFError:
        # 按 Ctrl+D（macOS / Linux）或 Ctrl+Z 再回车（Windows）时输入流结束，
        # input() 会抛 EOFError，在这里接住当作正常退出
        print('\n会话已结束，欢迎下次再来')
        break
    # strip + lower：容忍大小写和首尾空格，Quit、EXIT 都能退出
    if user_input.strip().lower() in EXIT_WORDS:
        print('会话已结束，欢迎下次再来')
        break
    messages.append(HumanMessage(content=user_input))
    # 送模型的是裁剪后的副本，messages 本身保持完整——
    # 两份数据用途不同：一份控 token，一份留档
    memory_messages = keep_recent_messages(messages, max_pairs=MAX_PAIRS_HISTORY)

    # 落盘放在「发送之前」，且存的是完整的 messages（不是 memory_messages）：
    # 这样文件里是全部历史；只请求过一次时内容就是 [system, human]，不会多出一条 ai
    history_path = save_history(messages, "01-messages03-history")
    print(f"历史已更新：{history_path}")

    # end="" + flush=True：让「小谷姐姐: 」和后面的流式内容接在同一行，
    # 且不换行就先刷出来，不等缓冲区满
    print('小谷姐姐: ', end="", flush=True)
    response = chat.stream(memory_messages)
    # 流式返回的是一串 AIMessageChunk，逐个拼接才是完整回复。
    # 首次直接用第一块，之后用 + 累加（AIMessageChunk 重载了 __add__）
    reply_content: AIMessageChunk | None = None
    for chunk in response:
        reply_content = chunk if reply_content is None else reply_content + chunk
        print(chunk.text, end='', flush=True)
    if reply_content is None:
        raise ValueError('no response from LLM')
    # 先把响应并入历史，再宣布本轮结束——这样打印「第 i 轮结束」时，
    # messages 里已经含了本轮的问答，状态和文案对得上
    messages.append(reply_content)
    print("\n", "=" * 10, f"第{i}轮对话结束", "=" * 10, "\n")
    i += 1
