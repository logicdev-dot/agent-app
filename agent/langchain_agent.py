import os
import base64
from dotenv import load_dotenv

load_dotenv()

print("✅ Environment loaded.")

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")

if not MINIMAX_API_KEY:
    print("❌ Minimax API key not found!")
    exit(1)

print("✅ Minimax API configured successfully.")

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from tools import (
    get_current_time,
    get_current_date,
    calculator,
    calculator_percentage,
    extract_entities,
    extract_attributes,
    analyze_text_structure,
    semantic_understanding,
    knowledge_retrieval,
    task_decomposition,
    plan_steps,
    analyze_requirements,
)

print("✅ Tools imported.")

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
    timeout=60,
)

print("✅ LangChain LLM created.")

tools = [
    get_current_time,
    get_current_date,
    calculator,
    calculator_percentage,
    task_decomposition,
    plan_steps,
]

print(f"✅ {len(tools)} tools loaded: {[t.name for t in tools]}")

system_prompt = """你是一个智能助手，能够使用工具来回答用户的问题。

可用工具功能：
1. 时间日期：获取当前时间和日期
2. 数学计算：进行数学运算和百分比计算
3. 特征识别：
   - 实体识别：从文本中提取人名、地名、组织名等
   - 属性提取：从文本中提取特定实体的属性信息
   - 文本结构分析：分析文本类型、主题、情感等
4. 知识理解：
   - 语义理解：理解用户查询的真正意图
   - 知识检索：基于知识库回答问题
5. 任务规划：
   - 任务分解：将复杂任务拆分成子任务
   - 步骤规划：制定详细的执行计划
   - 需求分析：提取和整理需求关键信息
6. 图片识别：分析图片内容

请根据用户的问题自动选择合适的工具来回答。
保持友好、专业的态度。"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

print("✅ Agent with tools created.")


def encode_image(image_path):
    """将图片转换为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_agent():
    print("\n🤖 智能助手已启动（增强版）")
    print("📌 可用能力：")
    print("   - 时间/日期查询")
    print("   - 数学计算（普通计算、百分比）")
    print("   - 特征识别（实体提取、属性提取、文本分析）")
    print("   - 知识理解（语义理解、知识检索）")
    print("   - 任务规划（任务分解、步骤规划、需求分析）")
    print("   - 图片识别")
    print("💡 输入 'exit' 退出，输入 'img:图片路径' 发送图片\n")

    chat_history = []

    while True:
        user_input = input("👤 输入: ")

        if user_input.lower() == 'exit':
            print("🤖 再见！")
            break

        if user_input.lower().startswith("img:"):
            image_path = user_input[4:].strip()
            if not os.path.exists(image_path):
                print(f"❌ 图片文件不存在: {image_path}")
                continue

            try:
                image_base64 = encode_image(image_path)
                content = [
                    {"type": "text", "text": "请描述这张图片的内容"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
                messages = chat_history + [HumanMessage(content=content)]
                result = agent.invoke({"messages": messages})
                chat_history = result.get("messages", [])

                for msg in reversed(chat_history):
                    if hasattr(msg, "content"):
                        if isinstance(msg.content, list):
                            for item in msg.content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    print(f"\n🤖 回复: {item.get('text')}\n")
                                    break
                        elif isinstance(msg.content, str):
                            print(f"\n🤖 回复: {msg.content}\n")
                        break
            except Exception as e:
                print(f"❌ 图片处理失败: {e}")
            continue

        messages = chat_history + [{"role": "user", "content": user_input}]
        result = agent.invoke({"messages": messages})

        chat_history = result.get("messages", [])

        for msg in reversed(chat_history):
            if hasattr(msg, "type") and msg.type == "human":
                continue
            if hasattr(msg, "type") and msg.type == "tool":
                continue
            if hasattr(msg, "content"):
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            print(f"\n🤖 回复: {item.get('text')}\n")
                            break
                elif isinstance(msg.content, str):
                    print(f"\n🤖 回复: {msg.content}\n")
                break


if __name__ == "__main__":
    run_agent()
