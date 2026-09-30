"""直接调用工具 —— 不经过模型，手动触发一次工具执行。

工具最终是给模型用的，但在交给模型之前，一般要先自己调一次，
确认参数收得对不对、返回值是不是想要的。这一步不需要模型参与，
拿着工具对象 invoke 就行，入参是参数字典。

后面几节的例子才是「把工具交给模型，由模型决定什么时候调」。

工具被 @tool 装饰后不再是普通函数，而是 StructuredTool 对象。
函数上的三样东西被映射成了工具的属性：

    函数名     → name
    docstring  → description（模型靠它判断该不该调这个工具）
    类型注解   → args_schema（参数校验用的 Pydantic 模型类）

文件末尾把这几个属性挨个打印出来，最后真正调一次。
"""

from typing import Any, cast

from langchain.tools import tool
from langchain_core.tools import BaseTool, StructuredTool


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"

    返回:
        天气信息字符串
    """
    return city + "晴天，温度 15°C"


# @tool 之后 get_weather 是 StructuredTool 对象，但 PyCharm 在 langchain
# 的 tool 重载里挑错了版本，把 @tool 的结果推成「装饰器函数」，
# 于是下面访问 .invoke 会报「找不到引用」。
# cast 只在类型层面生效、运行时原样返回，相当于手动告诉检查器
# 「这就是 StructuredTool」。套两层是因为单层直接转会改报「类型不兼容的转换」——
# 先转到最宽的 object 再往下转，两次转换各自都合法
get_weather = cast(StructuredTool, cast(object, get_weather))

# 确认上面那行 cast 没有改变运行时类型：装饰器造出来的就是 StructuredTool
# noinspection PyStringConversionWithoutDunderMethod
print(type(get_weather))
# StructuredTool 是 BaseTool 的子类，工具能被 Agent 调用靠的就是这个身份
print(isinstance(get_weather, BaseTool))

# 下面把「函数 → 工具」的三处映射挨个打印出来

# 函数名 → name
print('=' * 20, 'get_weather.name\n', get_weather.name)

# docstring → description：模型就是靠这句话判断该不该调这个工具，
# 所以工具函数的 docstring 写得清不清楚，直接影响模型会不会用、用得对不对
print('=' * 20, 'get_weather.description\n', get_weather.description)

# 类型注解 → args_schema：langchain 依据 city: str 自动生成的 Pydantic 模型类。
# 它的静态类型是 type[BaseModel] | dict[str, Any] | None 这个联合类型，
# 里面有成员没有 model_json_schema，PyCharm 会拦下来报「没有特性」；
# 运行时这里必定是那个模型类，所以 cast 到 Any 跳过检查
print('=' * 20, 'get_weather.args_schema\n', cast(Any, get_weather.args_schema).model_json_schema())

# 真正调一次。入参是参数字典，不是按函数签名那样写 city='北京'；
# 这里是直接调，不经过模型——工具本身就是一个可调用的对象
print('=' * 20, get_weather.invoke({"city": '北京'}))
