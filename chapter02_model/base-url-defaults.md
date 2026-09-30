# base_url 不配为什么也能跑 —— 环境变量与库默认值

## 起因

`01-model-openrouter01.py` 里有这么一行：

```python
print(os.environ["OPENROUTER_API_BASE"])
```

报 `KeyError: 'OPENROUTER_API_BASE'` —— 环境变量确实没配。
但同一个脚本里 `ChatOpenRouter` 却能正常发出请求。看似矛盾，其实不矛盾。

## 结论：两条完全独立的取值路径

| | 取值方式 | 环境变量缺失时 |
| --- | --- | --- |
| 脚本里的 `os.environ["X"]` | 直接下标访问 | 抛 `KeyError` |
| `ChatOpenRouter` 自己读 | `from_env("X", default=None)` | 拿到 `None`，不报错 |

`os.environ[...]` 是直接问操作系统要，没有就抛异常；
`from_env(..., default=None)` 问的是同一个问题，但**带了兜底**。
库有兜底、脚本没有 —— 区别只在这一句。

## 取值链路（OpenRouter）

字段定义在 `langchain_openrouter/chat_models.py:174`：

```python
openrouter_api_base: str | None = Field(
    default_factory=from_env("OPENROUTER_API_BASE", default=None),
    alias="base_url",          # 注释里写着：Maps to SDK `server_url`
)
```

环境变量没配 → 得到 `None` → 透传给 SDK 的 `server_url` → SDK 自己再判一次
（`openrouter/sdkconfiguration.py:49-56`）：

```python
if self.server_url is not None and self.server_url:
    return remove_suffix(self.server_url, "/"), {}
if not self.server:
    self.server = SERVER_PRODUCTION     # "https://openrouter.ai/api/v1"
```

兜底地址就在同文件第 22 行写死着：

```python
SERVER_PRODUCTION = "production"
SERVERS = {SERVER_PRODUCTION: "https://openrouter.ai/api/v1"}
```

所以请求发得出去，走的是 SDK 内置的官方端点，跟本地环境变量无关。

`OPENROUTER_API_BASE` 是**可选配置**，用途是指向别的端点（区域端点
`eu.openrouter.ai/api/v1`、自建代理等），不配就用官方默认。

## 后半段呢：`/chat/completions` 从哪来

上面兜底出来的是 `https://openrouter.ai/api/v1`，但实际请求发往
`https://openrouter.ai/api/v1/chat/completions`。这个后缀不在 base_url 里，
是 **SDK 中每个方法自带的**。

`ChatOpenRouter` 底层用的是 `openrouter` SDK（不是 OpenAI SDK）：

```python
# langchain_openrouter/chat_models.py:162
client: Any = Field(default=None, exclude=True)
"""Underlying SDK client (`openrouter.OpenRouter`)."""
```

每个接口在 SDK 里各自写死了 path（`openrouter/chat.py:790-794`）：

```python
req = self._build_request(
    method="POST",
    path="/chat/completions",      # ← 后缀在这儿
    base_url=base_url,             # ← 上面兜底出来的服务器地址
    ...
)
```

最后在 `openrouter/utils/url.py` 拼起来：

```python
return remove_suffix(server_url, "/") + path
```

```
https://openrouter.ai/api/v1   +   /chat/completions
= https://openrouter.ai/api/v1/chat/completions
```

**一句话：base_url 是「服务器在哪」，path 是「接口在服务器上的位置」。**
前者你配，后者 SDK 定。

由此得出两条实用结论：

- **base_url 尾部不能带 `/chat/completions`** —— 会拼成
  `.../chat/completions/chat/completions`。`01-model-openrouter02.py:26`
  那句注释说的就是这个。
- **带不带结尾斜杠都能跑** —— `remove_suffix(server_url, "/")` 会先吃掉尾部多余的 `/`，
  不会拼出 `api/v1//chat/completions`。

换端点时（区域端点、自建代理）只需要换前半段，后缀仍由 SDK 补上。

## 同一个模式：Ollama

`03-model-ollama01.py` 里没传 `base_url`，照样能跑，道理一样：

```
ChatOllama.base_url 默认 None
    → ollama SDK 的 host=None
    → os.getenv('OLLAMA_HOST')          取不到
    → 兜底 http://127.0.0.1:11434
```

## 什么情况下必须显式传 base_url

- 端点不是官方默认（自建代理、区域端点）
- 同一个 SDK 要对接不同厂商的兼容端点
  （`01-model-deepseek02.py`、`01-model-zhipu02.py` 就是这种情况）
- 服务不在默认主机或端口（Ollama 跑在别的机器上）

## 一个仍在踩的坑

`03-model-ollama02.py:17` 仍然写着：

```python
BASE_URL = os.environ["OLLAMA_BASE_URL"]
```

`OLLAMA_BASE_URL` 既不在 `.env` 里，也没在 shell 里 export ——
和本文开头那个 `OPENROUTER_API_BASE` 是**同一个错误**，运行时必 KeyError。
`03-model-ollama01.py` 已经把这行去掉了，02 还留着。

## 判断方法

不要拿「环境变量配没配」来判断能不能跑，要看**库的默认值**：

1. 找到字段定义，看它是 `default=...` 还是 `from_env("X")`（无兜底）
2. 有默认值 → 不配也能跑，环境变量只是覆盖手段
3. 没有默认值 → 必填，不配就报错

## 相关文件

- `01-model-openrouter01.py` —— 不传 base_url，走 SDK 默认端点
- `01-model-openrouter02.py` —— 显式传 `base_url="https://openrouter.ai/api/v1"`
- `03-model-ollama01.py` —— 用 ChatOllama，不传 base_url
- `03-model-ollama02.py` —— 读 `OLLAMA_BASE_URL`，会 KeyError
