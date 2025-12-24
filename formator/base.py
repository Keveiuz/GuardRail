from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from utils import GuardrailDataset, GuardrailItem
from utils.logger import Logger


class BaseFormator(ABC):
    """
    Formator 基类。
    输入 GuardrailDataset，输出标准对话结构（列表形式）。
    子类需实现具体的模版填充与字段组织逻辑。
    """

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger()

    @abstractmethod
    def format(self, data: GuardrailDataset, split: str = "") -> List[Dict[str, Any]]:
        """
        将数据集格式化为标准对话结构。
        Args:
            data: 由 BaseDatasetLoader.load 返回的 GuardrailDataset。
            split: 数据来源或分片标识，便于下游追踪。
        Returns:
            FormattedDataset: 标准对话结构列表。
        """
        raise NotImplementedError

    @staticmethod
    def _normalize_label(value: Optional[int]) -> Optional[int]:
        """将标签安全转换为 int 或 None。"""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _label_to_text(value: Optional[int]) -> Optional[str]:
        """将 0/1/-1 转换为 harmful/unharmful 文本标签。"""
        normalized = BaseFormator._normalize_label(value)
        if normalized == 1:
            return "harmful"
        if normalized == 0:
            return "unharmful"
        return None

    @staticmethod
    def _refusal_to_text(value: Optional[int]) -> Optional[str]:
        """将 0/1 转换为 refusal/compliance 文本标签。"""
        normalized = BaseFormator._normalize_label(value)
        if normalized == 1:
            return "refusal"
        if normalized == 0:
            return "compliance"
        return None

    @staticmethod
    def _string_or_none(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _item_to_dict(item: GuardrailItem) -> Dict[str, Any]:
        """便于调试或下游处理的基础字典形式。"""
        return {
            "prompt": item.prompt,
            "response": item.response,
            "prompt_harm_label": item.prompt_harm_label,
            "response_harm_label": item.response_harm_label,
            "response_refusal_label": item.response_refusal_label,
            "category": item.category,
            "extra": item.extra,
        }
