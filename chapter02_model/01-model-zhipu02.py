import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from utils import save_response

# 读取 .env 文件并写入 os.environ
# override=True：.env 里的值覆盖已存在的同名环境变量（默认不覆盖）
load_dotenv(override=True)

# 下标取值而非 .get()：拿到的是 str 而不是 str | None，缺变量时当场 KeyError
MODEL = os.environ["ZHIPUAI_MODEL"]

# openai_api_key 声明为 SecretStr | None | Callable[...]，而 SecretStr 不是 str 的子类，
# 显式包一层，让运行时和类型检查器看到同一个类型（否则 PyCharm 会报 str 不能赋值）
# 附带好处：SecretStr 的 repr 固定为 **********，key 不会被日志或报错信息打印出来
API_KEY = SecretStr(os.environ["ZHIPUAI_API_KEY"])

# 创建智谱 LLM 实例
# 这条路和 01-model-zhipu01.py 是两种不同的鉴权方式：
# 这里走智谱的 OpenAI 兼容端点，api_key 直接当 Bearer token 用，
# 不需要 PyJWT 在本地签 JWT，也就没有 InsecureKeyLengthWarning
#
# base_url 只能写到 /v4：OpenAI SDK 会自己拼 /chat/completions；
# 而 ChatZhipuAI 的 api_base 必须填完整 endpoint —— 两者语义相反，写混会 404
llm_zhipu02 = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.6,
)

response = llm_zhipu02.invoke('一句话介绍下你自己')
print(response)

# 把完整的 AIMessage 落盘成 JSON，实现见 utils.save_response
# 第二个参数是文件名前缀，用脚本名，便于一眼看出产物是哪来的
out_path = save_response(response, "01-model-zhipu02")
print(f"已写入：{out_path}")
