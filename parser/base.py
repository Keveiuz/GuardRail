from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from utils import Logger


class BaseParser(ABC):
    """
    解析模型推理结果的基类。

    约定：
    - 输入：一组模型输出（通常是字符串列表）
    - 输出：与输入等长的标签信息列表，每个元素为 dict
    """

    def __init__(self, logger: Optional[Logger] = None) -> None:
        self.logger = logger or Logger()

    @abstractmethod
    def parse(self, outputs: List[str]) -> List[Dict[str, Any]]:
        """
        解析一组模型输出，抽取结构化标签。

        Args:
            outputs: 每个元素是一条模型输出的完整文本。

        Returns:
            List[Dict]: 与输入长度一致的列表，每个元素为标签字典。
        """
        raise NotImplementedError


