from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前日期和时间

    返回:
        当前日期和时间的字符串，格式为 YYYY-MM-DD HH:MM:SS
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_current_date() -> str:
    """获取当前日期

    返回:
        当前日期的字符串，格式为 YYYY-MM-DD
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d")
