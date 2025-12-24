from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from utils import GuardrailItem, GuardrailDataset

class BaseDatasetLoader(ABC):
    """所有数据集的共同接口"""

    @abstractmethod
    def load(self) -> GuardrailDataset:
        """
        返回符合 GuardrailItem 结构的数据列表
        
        Returns:
            GuardrailDataset: 符合 GuardrailItem 结构的数据列表
        """
        pass

    def make_item(
        self,
        prompt: str,
        response: str,
        prompt_harm_label: Optional[int],
        response_harm_label: Optional[int],
        response_refusal_label: Optional[int],
        category: Optional[str],
        extra: Optional[Dict[str, Any]] = None
    ) -> GuardrailItem:
        """
        创建符合 GuardrailItem 结构的数据项
        
        Args:
            prompt: 用户提示文本
            response: 模型响应文本
            prompt_harm_label: 提示有害标签 (0=安全, 1=有害, None会被转换为0)
            response_harm_label: 响应有害标签 (0=安全, 1=有害, None会被转换为0)
            response_refusal_label: 响应拒绝标签 (0=合规, 1=拒绝, None会被转换为0)
            category: 类别名称 (None会被转换为空字符串)
            extra: 额外信息字典
            
        Returns:
            GuardrailItem: 符合结构的数据项
        """
        return GuardrailItem( 
            prompt=str(prompt),
            response=str(response),
            prompt_harm_label=int(prompt_harm_label) if prompt_harm_label is not None else -1,
            response_harm_label=int(response_harm_label) if response_harm_label is not None else -1,
            response_refusal_label=int(response_refusal_label) if response_refusal_label is not None else -1,
            category=str(category) if category is not None else "",
            extra=extra if extra is not None else None
        )
