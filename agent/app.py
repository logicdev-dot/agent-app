import os
import streamlit as st
from dotenv import load_dotenv

os.environ["STREAMLIT_GATHER_USAGE_STATS"] = "false"

load_dotenv()

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from tools import (
    get_current_time,
    get_current_date,
    calculator,
    calculator_percentage,
    analyze_image_with_qwen,
    describe_image_qwen,
    analyze_image_base64,
    extract_entities,
    extract_attributes,
    analyze_text_structure,
    semantic_understanding,
    knowledge_retrieval,
    task_decomposition,
    plan_steps,
    analyze_requirements,
)

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
    timeout=60,
)

tools = [
    get_current_time,
    get_current_date,
    calculator,
    calculator_percentage,
    analyze_image_with_qwen,
    describe_image_qwen,
    analyze_image_base64,
    extract_entities,
    extract_attributes,
    analyze_text_structure,
    semantic_understanding,
    knowledge_retrieval,
    task_decomposition,
    plan_steps,
    analyze_requirements,
]

system_prompt = """你是一个智能助手，能够使用工具来回答用户的问题。

可用工具功能：
1. 时间日期：获取当前时间和日期
2. 数学计算：进行数学运算和百分比计算
3. 图片分析：分析图片内容（使用base64编码的图片数据）
4. 特征识别：
   - 实体识别：从文本中提取人名、地名、组织名等
   - 属性提取：从文本中提取特定实体的属性信息
   - 文本结构分析：分析文本类型、主题、情感等
5. 知识理解：
   - 语义理解：理解用户查询的真正意图
   - 知识检索：基于知识库回答问题
6. 任务规划：
   - 任务分解：将复杂任务拆分成子任务
   - 步骤规划：制定详细的执行计划
   - 需求分析：提取和整理需求关键信息

请根据用户的问题自动选择合适的工具来回答。
保持友好、专业的态度。"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

st.set_page_config(
    page_title="AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --color-primary: #4F46E5;
        --color-primary-hover: #4338CA;
        --color-primary-light: #6366F1;
        --color-secondary: #8B5CF6;
        --color-accent: #06B6D4;
        --color-bg: #F8F9FA;
        --color-bg-secondary: #FFFFFF;
        --color-bg-card: #FFFFFF;
        --color-bg-input: #F3F4F6;
        --color-text: #1A1A2E;
        --color-text-muted: #6B7280;
        --color-text-secondary: #4B5563;
        --color-border: #E5E7EB;
        --color-success: #10B981;
        --color-warning: #F59E0B;
        --color-error: #EF4444;
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-full: 9999px;
    }
    
    .stApp {
        background: var(--color-bg);
        color: var(--color-text);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--color-bg-secondary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--color-border);
        border-radius: var(--radius-full);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--color-text-muted);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--color-text);
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        font-size: 1rem;
        color: var(--color-text-muted);
        margin-bottom: 2rem;
    }
    
    .card {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
    }
    
    .tool-card {
        background: var(--color-bg-secondary);
        border-radius: var(--radius-md);
        padding: 0.875rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    
    .tool-card:hover {
        border-color: var(--color-primary);
        transform: translateX(4px);
        box-shadow: var(--shadow-sm);
    }
    
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: var(--radius-lg);
        margin-bottom: 1rem;
        max-width: 85%;
        line-height: 1.6;
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .chat-message-assistant {
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1.5rem 1rem;
        border-bottom: 1px solid var(--color-border);
        margin-bottom: 1rem;
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    
    .stButton > button {
        background: var(--color-primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--color-primary-hover);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    .stTextInput > div > div {
        background: var(--color-bg-input);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        color: var(--color-text);
    }
    
    .stTextInput > div > div:focus-within {
        border-color: var(--color-primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
    
    .stFileUploader > div {
        background: var(--color-bg-secondary);
        border: 2px dashed var(--color-border);
        border-radius: var(--radius-md);
        padding: 1.5rem;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--color-primary);
    }
    
    .streamlit-expanderHeader {
        background: var(--color-bg-secondary);
        border-radius: var(--radius-md);
        color: var(--color-text-secondary);
    }
    
    .stat-card {
        background: var(--color-bg-secondary);
        border-radius: var(--radius-md);
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--color-primary-light);
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: var(--color-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    * {
        transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
    }
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thinking_steps" not in st.session_state:
    st.session_state.thinking_steps = []

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

if "image_processed" not in st.session_state:
    st.session_state.image_processed = True

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">✨</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #1A1A2E;">AI Assistant</div>
        <div style="font-size: 0.8rem; color: #6B7280;">Powered by Claude</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-title">🛠️ Available Tools</div>', unsafe_allow_html=True)
    
    tools_info = {
        "Time & Date": [
            ("🕐", "get_current_time", "Get current time"),
            ("📅", "get_current_date", "Get current date")
        ],
        "Calculator": [
            ("🧮", "calculator", "Math calculations"),
            ("📊", "calculator_percentage", "Percentage")
        ],
        "Vision (Qwen)": [
            ("🖼️", "analyze_image_with_qwen", "Analyze with Qwen VL"),
            ("📷", "describe_image_qwen", "Describe image (Qwen)")
        ],
        "Feature Recognition": [
            ("🏷️", "extract_entities", "Extract entities"),
            ("📋", "extract_attributes", "Extract attributes"),
            ("📝", "analyze_text_structure", "Analyze text")
        ],
        "Knowledge": [
            ("🧠", "semantic_understanding", "Understand intent"),
            ("📚", "knowledge_retrieval", "Knowledge search")
        ],
        "Task Planning": [
            ("📌", "task_decomposition", "Decompose task"),
            ("📑", "plan_steps", "Plan steps"),
            ("📊", "analyze_requirements", "Analyze requirements")
        ]
    }
    
    for category, items in tools_info.items():
        with st.expander(f"{category}", expanded=False):
            for icon, tool_name, tool_desc in items:
                st.markdown(f"""
                <div class="tool-card">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 1rem;">{icon}</span>
                        <div>
                            <div style="font-weight: 500; color: #1F2937; font-size: 0.875rem;">{tool_name}</div>
                            <div style="font-size: 0.75rem; color: #6B7280;">{tool_desc}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.thinking_steps = []
        st.session_state.image_processed = True
        st.rerun()

st.markdown("""
<div class="main-title">Hello, I'm Your AI Assistant</div>
<div class="subtitle">I can help you answer questions, analyze images, and use various tools</div>
""", unsafe_allow_html=True)

col_main, col_info = st.columns([3, 1])

with col_main:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], caption="Uploaded image", width=200)
                if msg.get("content"):
                    st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(msg["content"])

    col_input, col_upload = st.columns([3, 1])
    
    with col_input:
        with st.form("chat_form", clear_on_submit=True):
            col_text, col_btn = st.columns([4, 1])
            with col_text:
                prompt = st.text_input("输入文字...", label_visibility="collapsed", placeholder="输入文字或上传图片...")
            with col_btn:
                send_btn = st.form_submit_button("发送", use_container_width=True)
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "📎", 
            type=["jpg", "jpeg", "png", "gif", "bmp", "webp"], 
            key="file_uploader"
        )

    if send_btn and (uploaded_file or prompt):
        has_image = uploaded_file is not None and uploaded_file.name != st.session_state.get("last_uploaded_file", "")
        has_text = prompt and prompt.strip()
        
        if has_image:
            st.session_state.last_uploaded_file = uploaded_file.name
        
        import base64
        base64_image = None
        if uploaded_file:
            base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")
        
        current_text = prompt if prompt else ""
        
        with st.chat_message("user", avatar="👤"):
            if uploaded_file:
                st.image(uploaded_file, caption="上传的图片", width=200)
            if current_text:
                st.markdown(current_text)

        st.session_state.messages.append({
            "role": "user",
            "content": current_text if current_text else "[图片分析]",
            "image": uploaded_file
        })

        if has_image:
            from tools.vision_tool import analyze_image_with_qwen
            
            with st.spinner("🔍 分析中..."):
                try:
                    result = analyze_image_with_qwen.invoke({
                        "image_data": base64_image,
                        "media_type": uploaded_file.type,
                        "user_question": current_text
                    })
                    if current_text:
                        assistant_response = result
                    else:
                        assistant_response = result
                except Exception as e:
                    assistant_response = f"图片分析出错: {str(e)}"
        else:
            messages = st.session_state.chat_history + [{"role": "user", "content": current_text}]
            
            with st.spinner("🤔 思考中..."):
                result = agent.invoke({"messages": messages})
            
            st.session_state.chat_history = result.get("messages", [])
            
            messages_list = st.session_state.chat_history
            
            thinking_steps = []
            for i, msg in enumerate(messages_list):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        thinking_steps.append({
                            "type": "tool_call",
                            "name": tc.get("name", "unknown"),
                            "args": tc.get("args", {}),
                        })

                if hasattr(msg, "type") and msg.type == "tool":
                    thinking_steps.append({
                        "type": "tool_result",
                        "name": msg.name,
                        "content": msg.content,
                    })
            
            assistant_response = ""
            for msg in reversed(messages_list):
                if hasattr(msg, "type") and msg.type == "human":
                    continue
                if hasattr(msg, "type") and msg.type == "tool":
                    continue
                if hasattr(msg, "content"):
                    if isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                assistant_response = item.get("text", "")
                                break
                        if assistant_response:
                            break
                    elif isinstance(msg.content, str):
                        assistant_response = msg.content
                        break
            
            if not assistant_response:
                assistant_response = "Done"
            
            st.session_state.thinking_steps = thinking_steps

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(assistant_response)
        
        if has_image:
            st.session_state.thinking_steps = [{
                "type": "tool_result",
                "name": "analyze_image_with_qwen",
                "content": assistant_response
            }]

with col_info:
    st.markdown('<div class="sidebar-title">🧠 Thinking Process</div>', unsafe_allow_html=True)
    
    if st.session_state.thinking_steps:
        for i, step in enumerate(st.session_state.thinking_steps):
            if step["type"] == "tool_call":
                with st.expander(f"🔧 {step['name']}", expanded=False):
                    st.json(step.get("args", {}))
            elif step["type"] == "tool_result":
                with st.expander(f"📤 {step['name']}", expanded=False):
                    st.code(step.get("content", "")[:500])
    else:
        st.info("No thinking process yet")
    
    st.markdown("---")
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{len(st.session_state.messages)}</div>
        <div class="stat-label">Messages</div>
    </div>
    """, unsafe_allow_html=True)
