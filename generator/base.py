"""
模型生成接口基类
要求：输入 query 列表，输出 GenerationResult 列表（长度与顺序一致）
"""
from abc import ABC, abstractmethod
from typing import List


class BaseGenerator(ABC):
    """统一的模型生成接口"""

    @abstractmethod
    def generate(self, queries: List[str]) -> List[str]:
        """
        执行批量推理

        Args:
            queries: 待推理的字符串列表

        Returns:
            GenerationList: 与输入等长、顺序对应的生成结果列表
        """
        raise NotImplementedError

    def generate_one(self, query: str) -> str:
        """
        便捷方法：单条推理
        """
        return self.generate([query])[0]

