import re
from typing import List, Dict, Any, Optional

from utils import Logger

from .base import BaseParser


class WildGuardParser(BaseParser):
    """
    解析基于 WildGuard 模板的模型输出，抽取三个标签：
    - prompt_harm_label: 0=unharmful, 1=harmful, -1=未解析
    - response_refusal_label: 0=compliance, 1=refusal, -1=未解析/null
    - response_harm_label: 0=unharmful, 1=harmful, -1=未解析/null

    输入假定包含类似：

        Answers:
        Request: harmful
        Completion: refusal
        Response: unharmful

    逻辑与 `filter/confidence_filter.py` 中的 `extract_labels_from_text` 保持一致。
    """

    # 与 ConfidenceFilter 中相同的模式：允许可选空格、大小写、none/null
    _answer_pattern = re.compile(
        r"Request\s*[:\-]?\s*(harmful|unharmful|none|null)\b.*?"
        r"Completion\s*[:\-]?\s*(refusal|compliance|none|null)\b.*?"
        r"Response\s*[:\-]?\s*(harmful|unharmful|none|null)\b",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, logger: Optional[Logger] = None) -> None:
        super().__init__(logger=logger)

    @staticmethod
    def _map_harm(value: str) -> int:
        v = value.lower().strip()
        if v in ["harmful"]:
            return 1
        if v in ["unharmful", "harmless", "not harmful"]:
            return 0
        if v in ["none", "null", "empty"]:
            return -1
        return -1

    @staticmethod
    def _map_refusal(value: str) -> int:
        v = value.lower().strip()
        if v == "refusal":
            return 1
        if v == "compliance":
            return 0
        if v in ["none", "null", "empty"]:
            return -1
        return -1

    def _parse_single(self, text: str, idx: int = 0) -> Dict[str, Any]:
        """
        解析单条输出。
        """
        # 去掉简单 Markdown 粗体，避免 **harmful** 影响正则
        cleaned = text.replace("**", "")
        m = self._answer_pattern.search(cleaned)
        if not m:
            # self.logger.warn(
            #     f"[WildGuardParser] 第 {idx} 条输出未匹配到 Answers 段落，将返回全 -1 标签"
            # )
            return {
                "prompt_harm_label": -1,
                "response_refusal_label": -1,
                "response_harm_label": -1,
            }

        prompt_harm_label = self._map_harm(m.group(1))
        response_refusal_label = self._map_refusal(m.group(2))
        response_harm_label = self._map_harm(m.group(3))

        return {
            "prompt_harm_label": prompt_harm_label,
            "response_refusal_label": response_refusal_label,
            "response_harm_label": response_harm_label,
        }

    def parse(self, outputs: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i, text in enumerate(outputs):
            if not isinstance(text, str):
                self.logger.warn(
                    f"[WildGuardParser] 第 {i} 条输出不是字符串类型，实际类型 {type(text)}，将返回全 -1 标签"
                )
                results.append(
                    {
                        "prompt_harm_label": -1,
                        "response_refusal_label": -1,
                        "response_harm_label": -1,
                    }
                )
                continue
            results.append(self._parse_single(text, idx=i))
        return results


