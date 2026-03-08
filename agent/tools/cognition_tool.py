from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
import os
from dotenv import load_dotenv

load_dotenv()

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic")

llm = ChatAnthropic(
    model="claude-3-opus-20240229",
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
    timeout=60,
)


@tool
def extract_entities(text: str) -> str:
    """
    从文本中提取命名实体（人名、地名、组织名等）。
    当用户需要从文本中识别和提取关键实体时使用。
    """
    prompt = f"""请从以下文本中提取所有命名实体，并按类别分类：
    
文本：{text}

请以以下格式输出：
- 人名：[列表]
- 地名：[列表]  
- 组织名：[列表]
- 其他重要实体：[列表]

如果某个类别没有实体，请写"无"。"""

    result = llm.invoke(prompt)
    return result.content


@tool
def extract_attributes(text: str, target_entity: str) -> str:
    """
    从文本中提取特定实体的属性信息。
    当用户需要从文本中提取某个实体/事物的属性特征时使用。
    
    参数：
    - text: 要分析的文本
    - target_entity: 目标实体名称
    """
    prompt = f"""请从以下文本中提取"{target_entity}"的所有属性信息：

文本：{text}

请列出该实体的所有属性，包括但不限于：
- 基本特征
- 行为动作
- 关系描述
- 其他相关信息

如果文本中没有关于该实体的信息，请说明"未找到相关信息"。"""

    result = llm.invoke(prompt)
    return result.content


@tool
def analyze_text_structure(text: str) -> str:
    """
    分析文本的结构和特征。
    当用户需要理解文本的组成结构、主题分类或内容概览时使用。
    """
    prompt = f"""请分析以下文本的结构和特征：

文本：{text}

请提供：
1. 文本类型（新闻、对话、说明文等）
2. 主题/话题
3. 关键要点（3-5个）
4. 情感倾向（正面/中性/负面）
5. 文本结构分析"""

    result = llm.invoke(prompt)
    return result.content


@tool
def semantic_understanding(query: str, context: str = "") -> str:
    """
    理解用户查询的语义意图。
    当用户需要理解复杂问题的真正意图，或需要将模糊问题转化为具体问题时使用。
    
    参数：
    - query: 用户的查询/问题
    - context: 可选的上下文信息
    """
    context_hint = f"\n\n参考上下文：{context}" if context else ""
    
    prompt = f"""请分析以下查询的语义意图：{context_hint}

查询：{query}

请提供：
1. 用户真正的意图是什么
2. 问题涉及的关键概念
3. 可能的隐含问题
4. 最佳回答方向建议"""

    result = llm.invoke(prompt)
    return result.content


@tool
def knowledge_retrieval(question: str, knowledge_base: str = "") -> str:
    """
    基于问题检索相关知识。
    当用户询问需要特定领域知识的问题时使用。
    
    参数：
    - question: 用户的问题
    - knowledge_base: 可选的知识库内容，如果没有则基于LLM自身知识回答
    """
    if knowledge_base:
        prompt = f"""基于以下知识库内容回答用户问题：

知识库：
{knowledge_base}

问题：{question}

请根据知识库内容给出准确回答。如果知识库中没有相关信息，请说明"根据提供的信息无法回答"。"""
    else:
        prompt = f"""请回答以下问题，你可以运用你的知识库：

问题：{question}

请给出准确、全面的回答。"""

    result = llm.invoke(prompt)
    return result.content


@tool
def task_decomposition(task: str) -> str:
    """
    将复杂任务分解为可执行的子任务。
    当用户提出一个复杂或多步骤的任务时使用。
    
    参数：
    - task: 要分解的任务描述
    """
    prompt = f"""请将以下任务分解为具体的子任务步骤：

任务：{task}

请按执行顺序列出：
1. 第一步子任务
2. 第二步子任务
3. ...

每个子任务应该：
- 描述具体要做什么
- 说明完成标准
- 标注依赖关系（如需要前置步骤）"""

    result = llm.invoke(prompt)
    return result.content


@tool
def plan_steps(goal: str, constraints: str = "") -> str:
    """
    规划实现目标的详细步骤。
    当用户需要制定详细的执行计划时使用。
    
    参数：
    - goal: 要达成的目标
    - constraints: 可选的约束条件
    """
    constraint_hint = f"\n约束条件：{constraints}" if constraints else ""
    
    prompt = f"""请为以下目标制定详细的执行计划：

目标：{goal}{constraint_hint}

请提供：
1. 总体计划框架
2. 具体执行步骤（按顺序）
3. 每步骤的预计时间
4. 可能遇到的问题及应对方案
5. 最终验收标准"""

    result = llm.invoke(prompt)
    return result.content


@tool
def analyze_requirements(requirement_text: str) -> str:
    """
    分析需求并提取关键信息。
    当用户提供一段需求描述，需要提取和整理关键需求点时使用。
    """
    prompt = f"""请分析以下需求描述，提取关键信息：

需求：{requirement_text}

请提供：
1. 核心需求（必须满足）
2. 次要需求（最好满足）
3. 隐含需求（可能需要）
4. 需求优先级建议
5. 可能的冲突点（如果有）"""

    result = llm.invoke(prompt)
    return result.content
