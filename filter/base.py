from abc import ABC, abstractmethod
from typing import List

from utils import GenerationList


class BaseFilter(ABC):
    """
    过滤器基类：输入 GenerationList，输出过滤后的 GenerationList。
    """

    @abstractmethod
    def apply(self, generations: GenerationList) -> GenerationList:
        """
        子类实现具体过滤逻辑。
        """
        raise NotImplementedError

