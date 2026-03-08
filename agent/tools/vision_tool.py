import base64
import json
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_ENDPOINT = os.getenv("QWEN_API_ENDPOINT", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")


@tool
def analyze_image_with_qwen(image_data: str, media_type: str = "image/jpeg", user_question: str = "") -> str:
    """使用千问视觉模型分析图片内容

    参数:
        image_data: base64编码的图片数据
        media_type: 图片的MIME类型，如 image/jpeg, image/png, image/gif, image/webp
        user_question: 用户的问题（可选）

    返回:
        对图片内容的详细描述和分析
    """
    try:
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "disable"
        }
        
        if media_type == "image/jpeg":
            image_format = "jpeg"
        elif media_type == "image/png":
            image_format = "png"
        elif media_type == "image/gif":
            image_format = "gif"
        elif media_type == "image/webp":
            image_format = "webp"
        else:
            image_format = "jpeg"
        
        if user_question and user_question.strip():
            prompt_text = user_question
        else:
            prompt_text = "请详细描述这张图片的内容，包括场景、人物、物体、颜色等所有可见元素。用中文回答。"
        
        payload = {
            "model": "qwen-vl-plus",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"data:image/{image_format};base64,{image_data}"
                            },
                            {
                                "text": prompt_text
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "max_tokens": 1024,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            QWEN_API_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result and "choices" in result["output"]:
                choices = result["output"]["choices"]
                if len(choices) > 0 and "message" in choices[0]:
                    content = choices[0]["message"].get("content", [])
                    if len(content) > 0 and "text" in content[0]:
                        return content[0]["text"]
            if "output" in result and "text" in result["output"]:
                return result["output"]["text"]
            return f"结果: {json.dumps(result, ensure_ascii=False)}"
        else:
            return f"API错误: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"图片分析失败: {str(e)}"


@tool
def describe_image_qwen(image_base64: str) -> str:
    """使用千问模型描述图片

    参数:
        image_base64: base64编码的图片数据

    返回:
        图片的详细描述
    """
    return analyze_image_with_qwen.invoke({"image_data": image_base64, "media_type": "image/jpeg"})


@tool
def analyze_image_base64(image_data: str, media_type: str = "image/jpeg") -> str:
    """分析图片内容（使用base64编码的图片数据）

    参数:
        image_data: base64编码的图片数据
        media_type: 图片的MIME类型，如 image/jpeg, image/png, image/gif, image/webp

    返回:
        对图片内容的详细描述和分析
    """
    return analyze_image_with_qwen.invoke({"image_data": image_data, "media_type": media_type})
