import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr
from rich import print as rprint


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

# 支持的完整参数列表
print(ChatDeepSeek.model_fields.keys())
llm_deepseek = ChatDeepSeek(model=MODEL, api_key=API_KEY,max_tokens=1000,max_retries=6)
# profile 是这个模型的能力清单：上下文窗口多大、单次最多输出多少、支不支持图片输入和工具调用等。
# 它来自集成包内置的元数据，不发请求就能读到，常用来在调用前判断这个模型能不能满足需求。
# 注意：不是所有三方集成都补了这份数据，没有时会是 None 或空值，属正常现象。
rprint(llm_deepseek.profile)