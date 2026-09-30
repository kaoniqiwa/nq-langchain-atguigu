"""各章节脚本共用的工具函数。

放在项目根目录（langchain-atguigu/）而不是某个章节目录里，供所有章节共用。

本模块的公开函数：

- save_response / save_history：把结果落盘成 JSON。
  前者文件名带时间戳，给单次响应留档、不覆盖；
  后者文件名固定，每轮覆盖更新，适合会话历史。
  两者都不只认消息——Pydantic 实例、普通 dict 及其列表也收，
  因为结构化输出（chapter06）拿到的就不是 BaseMessage。
- keep_recent_messages：裁剪历史消息，保留 system + 最近 N 轮，用于控制上下文长度。
- image_to_base64 / image_to_data_uri：把本地图片编码成 base64 数据或 data URL，
  供多模态消息使用。

导入方式：各章节脚本按需导入，例如 `from utils import save_response`。
这要求项目根目录在 sys.path 里——PyCharm 默认会把项目根加进 PYTHONPATH，
命令行直接 `python chapter02_model/xx.py` 时则不会（那时只有脚本所在目录在 sys.path），
需要在根目录用 `python -m chapter02_model.xx` 或在脚本里手动补 sys.path。
"""

import base64
import json
import mimetypes
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, message_to_dict
from pydantic import BaseModel

# 所有产物的输出目录。用模块级常量而不是每次现算，
# 是因为它只取决于本文件的位置，与调用方脚本放在哪、从哪个工作目录运行都无关
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _write_json(data: Any, out_path: Path) -> None:
    """把数据写成 JSON 文件，父目录不存在时自动创建。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False：中文按原样写入，而不是 \uXXXX 转义
    # default=str 兜底：message_to_dict 内部走的是 model_dump()（python 模式），
    # 万一 response_metadata 里混进 datetime 这类非 JSON 对象，转成字符串而不是抛 TypeError
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _to_jsonable(data: Any) -> Any:
    """把各类「结果」转成能直接交给 json.dump 的结构。

    列表 / 元组会逐项递归处理，所以 batch 的返回值也能直接落盘。
    """
    if isinstance(data, BaseMessage):
        # message_to_dict 返回 {"type": "ai", "data": {...}}，
        # 和 sample-01/node/packages/model/src/writeToJson.ts 的产物结构一致，方便跨语言对照
        return message_to_dict(data)
    if isinstance(data, BaseModel):
        # with_structured_output(Person) 返回的是 Pydantic 实例，
        # 先转成普通 dict；否则 json.dump 会拿着 default=str 把它整个变成一个字符串
        return data.model_dump()
    if isinstance(data, Mapping):
        # Agent 的返回值是 {"messages": [HumanMessage, AIMessage, ...]} 这种状态字典。
        # 键原样保留，值继续往下递归——不递归的话，里面的消息对象同样会被
        # default=str 变成一串不可解析的字符串
        return {key: _to_jsonable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_jsonable(item) for item in data]
    # str、int 这些本来就能直接 dump，原样返回
    return data


# save_response 能落盘的结果类型：单条消息、Pydantic 实例、普通 dict，或它们的序列
Saveable = BaseMessage | BaseModel | Mapping[str, Any] | Sequence[Any]


def save_response(response: Saveable, prefix: str) -> Path:
    """把模型返回的结果落盘成 JSON，返回写入的文件路径。

    落盘的东西是什么，取决于调用它的场景：

        BaseMessage      普通对话，chat.invoke() 的返回值
        Pydantic 实例     with_structured_output(Person) 的返回值
        dict             with_structured_output(TypedDict) 的返回值，
                         或 Agent 的状态字典 {"messages": [...]}

    单个对象落盘成一个 JSON 对象，列表（batch 的返回值）落盘成一个 JSON 数组。

    文件名带时间戳，多次运行不会互相覆盖——适合给「某一次响应」留档。
    要持续更新同一个文件（比如会话历史），用 save_history。

    Args:
        response: 要落盘的结果，单个或列表都行。
        prefix: 输出文件名前缀，一般传调用方脚本名，例如 "01-model-deepseek"。
    """
    out_path = OUTPUT_DIR / f"{prefix}-{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    _write_json(_to_jsonable(response), out_path)
    return out_path


def save_history(messages: Sequence[BaseMessage], name: str) -> Path:
    """把整段消息历史覆盖写入 output/<name>.json，返回写入的文件路径。

    与 save_response 的区别：文件名固定、不带时间戳，每次调用覆盖同一份文件。
    适合会话过程中持续更新「当前历史」，文件里始终是最新的完整消息列表；
    代价是上一次的内容被覆盖，要保留每次的快照得用 save_response。

    Args:
        messages: 完整的消息列表。
        name: 文件名（不含扩展名），一般传脚本名加 "-history"，例如 "01-messages03-history"。
    """
    out_path = OUTPUT_DIR / f"{name}.json"
    _write_json(_to_jsonable(messages), out_path)
    return out_path


def keep_recent_messages(
        history: Sequence[BaseMessage], *, max_pairs: int = 3
) -> list[BaseMessage]:
    """保留 system 消息 + 最近的 N 轮对话，用于控制上下文长度。

    注意：实现是按下标按「条数」切，不是严格按「轮」切。
    消息严格交替（human/ai/human/ai…）时两者等价；但若出现连续两条 human，
    或列表里混进 ToolMessage，切出来可能不是完整的轮，
    甚至把 AIMessage(tool_calls) 和它对应的 ToolMessage 拆散，导致 API 报错。
    真要用在工具调用场景，得改成识别「轮」的边界来切。

    Args:
        history: 完整的历史消息列表。
        max_pairs: 保留的轮数，一轮 = human + ai 两条。

    Returns:
        新的消息列表，system 消息置顶，其后是最近 max_pairs 轮对话。
    """
    system_messages = [m for m in history if m.type == "system"]
    dialog = [m for m in history if m.type != "system"]
    # max_pairs=0 要显式返回空：切片 dialog[-0:] 等价于 dialog[0:]，会取到整个列表，
    # 因为 -0 == 0，这是负数索引的边界坑
    recent = dialog[-max_pairs * 2:] if max_pairs > 0 else []
    return system_messages + recent


class ImageBase64(TypedDict):
    """图片的 base64 数据和 MIME 类型。

    字段名刻意和 langchain_core 的 ImageContentBlock 保持一致，
    所以 image_to_base64() 的结果可以直接展开进 content block：
    {"type": "image", **image_to_base64("cat.jpg")}
    """

    base64: str
    mime_type: str


def _read_image(image_path: Path | str) -> tuple[str, str]:
    """读图片文件，返回 (base64 字符串, MIME 类型)。"""
    path = Path(image_path)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    # 扩展名推断不出来时兜底成 image/jpeg，否则拼出来是 "data:None;base64,..."
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    return b64, mime_type


def image_to_base64(image_path: Path | str) -> ImageBase64:
    """把本地图片编码成 base64 数据和 MIME 类型。

    返回 {"base64": ..., "mime_type": ...}，
    适合把图片数据和 MIME 拆成两个独立字段的接口，或直接构造 content block：

        HumanMessage(content_blocks=[
            {"type": "text", "text": "这张图里有什么？"},
            {"type": "image", **image_to_base64("cat.jpg")},
        ])
    """
    b64, mime_type = _read_image(image_path)
    return {"base64": b64, "mime_type": mime_type}


def image_to_data_uri(image_path: Path | str) -> str:
    """把本地图片编码成 data URL 字符串。

    返回形如 data:image/jpeg;base64,/9j/4AAQ... 的完整字符串，
    直接塞进 OpenAI 风格的 image_url.url：

        HumanMessage(content=[
            {"type": "text", "text": "这张图里是什么？"},
            {"type": "image_url", "image_url": {"url": image_to_data_uri("cat.jpg")}},
        ])
    """
    b64, mime_type = _read_image(image_path)
    return f"data:{mime_type};base64,{b64}"
