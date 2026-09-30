import os
import warnings

from dotenv import load_dotenv
from jwt import InsecureKeyLengthWarning
from langchain_community.chat_models import ChatZhipuAI

from utils import save_response

# 智谱的 key 格式是 {id}.{secret}，secret 段固定 16 字符；
# ChatZhipuAI 拿它当 HS256 的 HMAC 密钥，短于 RFC 7518 要求的 32 字节，
# 所以 PyJWT 每次调用都会告警。这是智谱 key 设计的固有限制，不是代码问题，
# 服务端也只验签、不校验密钥长度，因此这里直接屏蔽
warnings.filterwarnings("ignore", category=InsecureKeyLengthWarning)

# 读取 .env 文件并写入 os.environ
# override=True：.env 里的值覆盖已存在的同名环境变量（默认不覆盖）
load_dotenv(override=True)

# 下标取值而非 .get()：拿到的是 str 而不是 str | None，缺变量时当场 KeyError
MODEL = os.environ["ZHIPUAI_MODEL"]
API_KEY = os.environ["ZHIPUAI_API_KEY"]

# api_base 不传，用默认值 https://open.bigmodel.cn/api/paas/v4/chat/completions
# 注意：ChatZhipuAI 的 api_base 要填「完整 endpoint」，它直接 POST 到这个地址；
# 而 OpenAI 兼容客户端的 base_url 只需写到 /v4 —— 两者语义相反，写混会 404
llm_zhipu = ChatZhipuAI(model=MODEL, api_key=API_KEY)

response = llm_zhipu.invoke('一句话介绍下你自己')
print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-model-zhipu01")
print(f"已写入：{out_path}")
