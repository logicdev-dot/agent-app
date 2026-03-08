import math
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """数学计算器，执行基本的数学运算

    参数:
        expression: 数学表达式，如 "2+2", "sqrt(16)", "sin(3.14159/2)" 等

    返回:
        计算结果的字符串
    """
    try:
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def calculator_percentage(value: float, percentage: float) -> str:
    """计算百分比

    参数:
        value: 原始数值
        percentage: 百分比值，如 20 表示 20%

    返回:
        百分比计算结果
    """
    result = value * (percentage / 100)
    return f"{value} 的 {percentage}% = {result}"
