import re
from typing import List, Dict, Any, Optional, Union

from utils import Logger
from .base import BaseParser


class AegisV1SplitParser(BaseParser):
    """
    抽取 USER / ASSISTANT 对话结构。

    - 单轮 → {"query": str|None, "response": str|None}
    - 多轮 → {"query": [..], "response": [..]}
    """

    _turn_pattern = re.compile(
        r"\[TURN\s*(\d+)\]\s*ROLE\s*:\s*(USER|ASSISTANT)\s*TEXT\s*:\s*(.+?)(?=\n\s*\[TURN|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, logger: Optional[Logger] = None) -> None:
        super().__init__(logger=logger)

    def _parse_single(self, text: str, idx: int = 0) -> Dict[str, Any]:
        cleaned = text.strip()
        matches = self._turn_pattern.findall(cleaned)

        if not matches:
            self.logger.warn(
                f"[QueryResponseListParser] 第 {idx} 条输出未解析到任何 TURN"
            )
            return {"query": None, "response": None}

        # turn_id -> {"USER": ..., "ASSISTANT": ...}
        turns: Dict[str, Dict[str, Optional[str]]] = {}

        for turn_id, role, content in matches:
            role = role.upper()
            content = content.strip()
            if content.lower() == "none":
                content = None

            if turn_id not in turns:
                turns[turn_id] = {"USER": None, "ASSISTANT": None}

            turns[turn_id][role] = content

        # 按 turn_id 排序
        ordered_turns = [turns[k] for k in sorted(turns, key=lambda x: int(x))]

        queries: List[Optional[str]] = []
        responses: List[Optional[str]] = []

        for t in ordered_turns:
            queries.append(t.get("USER"))
            responses.append(t.get("ASSISTANT"))

        # 单轮 vs 多轮
        if len(queries) == 1:
            return {
                "query": queries[0],
                "response": responses[0],
            }

        return {
            "query": queries,
            "response": responses,
        }

    def parse(self, outputs: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i, text in enumerate(outputs):
            if not isinstance(text, str):
                self.logger.warn(
                    f"[QueryResponseListParser] 第 {i} 条输出不是字符串类型，实际类型 {type(text)}"
                )
                results.append({"query": None, "response": None})
                continue

            results.append(self._parse_single(text, idx=i))
        return results
