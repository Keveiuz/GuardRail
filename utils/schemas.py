"""
数据结构定义
定义数据交换的标准结构
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class GuardrailItem():
    """
    数据集项的标准结构
    
    每个数据项包含以下字段：
    - prompt: 用户提示文本 (字符串)
    - response: 模型响应文本 (字符串)
    - prompt_harm_label: 提示有害标签 (整数，0=安全, 1=有害)
    - response_harm_label: 响应有害标签 (整数，0=安全, 1=有害)
    - response_refusal_label: 响应拒绝标签 (整数，0=合规, 1=拒绝)
    - category: 类别名称 (字符串)
    - extra: 额外信息字典 (可选)
    """
    prompt: str
    response: str
    prompt_harm_label: int = -1
    response_harm_label: int = -1
    response_refusal_label: int = -1
    category: str = ""
    # extra 字段定义为可选，并提供默认值
    extra: Optional[Dict[str, Any]] = field(default_factory=lambda: None)


# 类型别名：数据集列表
GuardrailDataset = List[GuardrailItem]



@dataclass
class FormattedItem():
    """
    格式化数据项的标准结构
    
    每个数据项包含以下字段：
    - id: 每条数据的唯一标识
    - split: 每条数据所属的数据集
    - prompt: 用户提示文本 (字符串)
    - response: 模型响应文本 (字符串)
    - conversations: 格式化后的的对话信息（列表）
    - label: 模型判断的安全标签 (整数，0=安全, 1=有害)
    - prompt_harm_label: 提示有害标签 (整数，0=安全, 1=有害)
    - response_harm_label: 响应有害标签 (整数，0=安全, 1=有害)
    - response_refusal_label: 响应拒绝标签 (整数，0=合规, 1=拒绝)
    - category: 类别名称 (字符串)
    - extra: 额外信息字典 (可选)
    """
    id: int
    split: str
    prompt: str
    response: str
    conversations: List[Dict]
    label: int = -1
    prompt_harm_label: int = -1
    response_harm_label: int = -1
    response_refusal_label: int = -1
    category: str = ""
    # extra 字段定义为可选，并提供默认值
    extra: Optional[Dict[str, Any]] = field(default_factory=lambda: None)


# 类型别名：数据集列表
FormattedDataset = List[FormattedItem]


@dataclass
class GenerationCandidate():
    text: str
    # 由 logprobs 计算得到的整体 token 置信度（如平均 logprob）
    token_confidence: Optional[float] = None
    

@dataclass
class GenerationResult(FormattedItem):
    """
    推理结果项的标准结构
    
    每个数据项包含以下字段：
    - id: 每条数据的唯一标识（整数）
    - split: 每条数据所属的数据集（字符串）
    - prompt: 用户提示文本 (字符串)
    - response: 模型响应文本 (字符串)
    - conversations: 格式化后的的对话信息（列表）
    - candidates: 包含多个推理结果与置信度信息的列表（列表）
    - prompt_harm_label: 提示有害标签 (整数，0=安全, 1=有害)
    - response_harm_label: 响应有害标签 (整数，0=安全, 1=有害)
    - response_refusal_label: 响应拒绝标签 (整数，0=合规, 1=拒绝)
    - category: 类别名称 (字符串)
    - extra: 额外信息字典 (可选)
    """
    candidates: List[GenerationCandidate] = field(default_factory=list)

# 类型别名：生成结果列表
GenerationList = List[GenerationResult]


# ========== 过滤后数据结构 ==========
@dataclass
class FilteredItem():
    """
    过滤后数据项的标准结构
    
    每个数据项包含以下字段：
    - id: 每条数据的唯一标识
    - split: 每条数据所属的数据集
    - prompt: 用户提示文本 (字符串)
    - response: 模型响应文本 (字符串)
    - conversations: 需要判断的对话信息和模型的判断回复（列表）
    - confidence: 模型判断的置信度（浮点数）
    - label: 模型判断的安全标签 (整数，0=安全, 1=有害)
    - prompt_harm_label: 提示有害标签 (整数，0=安全, 1=有害)
    - response_harm_label: 响应有害标签 (整数，0=安全, 1=有害)
    - response_refusal_label: 响应拒绝标签 (整数，0=合规, 1=拒绝)
    - category: 类别名称 (字符串)
    """
    id: int
    split: str
    prompt: str
    response: str
    conversations: List[Dict]
    confidence: float
    label: int = -1
    prompt_harm_label: int = -1
    response_harm_label: int = -1
    response_refusal_label: int = -1
    category: str = ""


# 类型别名：过滤结果列表
FilteredDataset = List[FilteredItem]

