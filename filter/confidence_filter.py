import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from tabulate import tabulate
from tqdm import tqdm

from filter.base import BaseFilter
from utils import (
    ConfidenceCalculator,
    Logger,
    GenerationList,
    FilteredItem,
    FilteredDataset,
)


# ================== ⚙️ 配置类 ==================
@dataclass
class FilterConfig:
    # 输入 / 输出
    input_file: str = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-metric.jsonl"
    output_file: str = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-filtered.jsonl"

    # 筛选参数（两者都为 None 则只依据 score>=4）
    score_threshold: Literal[1, 2, 3, 4] = 4
    confidence_calc_method: Literal[
        "average_trace_confidence",
        "bottom_percent_group_confidence",
        "group_confidence",
        "lowest_group_confidence",
        "tail_confidence",
        "tail_confidence_by_percent",
    ] = "bottom_percent_group_confidence"
    confidence_threshold: Optional[float] = None
    select_ratio: Optional[float] = None
    select_n: Optional[int] = 10000
    length_control: Optional[float] = field(
        default_factory=lambda: len(
            "<think>\nAnswers:  \nRequest: unharmful  \nCompletion: compliance  \nResponse: unharmful\n</think>\n\n Request: unharmful \nCompletion: compliance \nResponse: unharmful"
        )
        + 50
    )

    # 新增：选择模式：'top' 或 'bottom'
    selection_mode: Literal["top", "bottom"] = "top"

    window_size: int = 1024
    bottom_percent: float = 0.1
    tail_size: int = 1024
    tail_percent: float = 0.1

    # 行为控制
    remove_token_confidence: bool = False
    num_candidates: int = 1


# ================== 工具函数 ==================
def draw_table(user_safe: int, user_unsafe: int, agent_safe: int, agent_unsafe: int):
    user_total = user_safe + user_unsafe
    agent_total = agent_safe + agent_unsafe
    safe_total = user_safe + agent_safe
    unsafe_total = user_unsafe + agent_unsafe
    grand_total = user_total + agent_total

    table_data = [
        ["user", user_safe, user_unsafe, user_total],
        ["agent", agent_safe, agent_unsafe, agent_total],
        ["", safe_total, unsafe_total, ""],
    ]
    colalign = ("center", "center", "center", "center")
    table_str = tabulate(table_data, headers=["", "safe", "unsafe", ""], tablefmt="grid", colalign=colalign)
    print(table_str)
    width = max(len(line) for line in table_str.split("\n"))
    total_str = f" total: {grand_total} "
    padding = width - 2 - len(total_str)
    left_pad = padding // 2
    right_pad = padding - left_pad
    print("|" + total_str + " " * (right_pad + left_pad) + "|")
    print("-" * width)


def extract_labels_from_text(text: str) -> Tuple[str, Optional[Dict[str, Optional[int]]], Optional[str]]:
    think_pattern = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
    think_match = think_pattern.search(text)
    think_content = think_match.group(1).strip() if think_match else None
    outside_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 去掉简单的 Markdown 粗体标记，方便正则匹配（如 **harmful**）
    outside_clean = outside_text.replace("**", "")
    answer_pattern = re.compile(
        r"Request\s*[:\-]?\s*(harmful|unharmful|none|null)\b.*?"
        r"Completion\s*[:\-]?\s*(refusal|compliance|none|null)\b.*?"
        r"Response\s*[:\-]?\s*(harmful|unharmful|none|null)\b",
        flags=re.IGNORECASE | re.DOTALL,
    )
    answer_match = answer_pattern.search(outside_clean)

    def map_harm(value: str) -> Optional[int]:
        v = value.lower().strip()
        if v in ["harmful"]:
            return 1
        if v in ["unharmful", "harmless", "not harmful"]:
            return 0
        if v in ["none", "null", "empty"]:
            return -1
        return -1

    def map_refusal(value: str) -> Optional[int]:
        v = value.lower().strip()
        if v == "refusal":
            return 1
        if v == "compliance":
            return 0
        if v in ["none", "null", "empty"]:
            return -1
        return -1

    if answer_match:
        prompt_harm = map_harm(answer_match.group(1))
        response_refusal = map_refusal(answer_match.group(2))
        response_harm = map_harm(answer_match.group(3))
        answer_dict = {
            "prompt_harm": prompt_harm,
            "response_harm": response_harm,
            "response_refusal": response_refusal,
        }
    else:
        answer_dict = None

    return outside_text, answer_dict, think_content


def compute_score(item: Dict[str, Any], answer_dict: Optional[Dict[str, Optional[int]]]) -> int:
    prompt_harm_label = item.get("prompt_harm_label")
    response_refusal_label = item.get("response_refusal_label")
    response_harm_label = item.get("response_harm_label")
    score = 0
    if answer_dict is None:
        return score
    score = 1
    if answer_dict.get("prompt_harm") != prompt_harm_label:
        return score
    score = 2
    if answer_dict.get("response_harm") != response_harm_label:
        return score
    score = 3
    if answer_dict.get("response_refusal") != response_refusal_label:
        return score
    score = 4
    return score


def _calc_confidence(cfg: FilterConfig, token_conf: Optional[List[float]]) -> Optional[float]:
    if token_conf is None:
        return None
    # 去除 None，确保非空
    filtered = [c for c in token_conf if c is not None]
    if len(filtered) == 0:
        return None

    calc = ConfidenceCalculator(token_confidences=filtered)
    if cfg.confidence_calc_method == "average_trace_confidence":
        return calc.compute_average_trace_confidence()
    if cfg.confidence_calc_method == "bottom_percent_group_confidence":
        return calc.compute_bottom_percent_group_confidence(window_size=cfg.window_size, bottom_percent=cfg.bottom_percent)
    if cfg.confidence_calc_method == "group_confidence":
        return calc.compute_group_confidence(window_size=cfg.window_size)
    if cfg.confidence_calc_method == "lowest_group_confidence":
        return calc.compute_lowest_group_confidence(window_size=cfg.window_size)
    if cfg.confidence_calc_method == "tail_confidence":
        return calc.compute_tail_confidence(tail_size=cfg.tail_size)
    if cfg.confidence_calc_method == "tail_confidence_by_percent":
        return calc.compute_tail_confidence_by_percent(tail_percent=cfg.tail_percent)

    raise TypeError(
        'unsupported confidence calculate method! supported: ["average_trace_confidence", '
        '"bottom_percent_group_confidence", "group_confidence", "lowest_group_confidence", '
        '"tail_confidence", "tail_confidence_by_percent"]'
    )


# ================== 主逻辑：类封装 ==================
class ConfidenceFilter(BaseFilter):
    """
    输入 GenerationList，输出过滤后的 GenerationList。
    """

    def __init__(self, cfg: Optional[FilterConfig] = None):
        self.cfg = cfg or FilterConfig()
        self.logger = Logger()

    @staticmethod
    def _get(item: Union[Dict[str, Any], Any], key: str, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def apply(self, generations: GenerationList) -> FilteredDataset:
        cfg = self.cfg
        logger = self.logger

        logger.info(f"Total queries loaded: {len(generations)}")
        query_map = {self._get(item, "id"): item for item in generations if self._get(item, "id") is not None}

        valid_candidates: List[Dict[str, Any]] = []
        valid_query_ids = set()

        cand_scores = {}
        cand_response = {}
        for item in tqdm(generations, desc="Validating candidates", unit="queries"):
            qid = item.id

            candidates: List[Dict[str, Any]] = self._get(item, "candidates", []) or []
            for cand in candidates[: cfg.num_candidates]:
                resp_text = cand.get("text", "") if isinstance(cand, dict) else getattr(cand, "text", "")

                cand_response[qid] = resp_text

                if cfg.length_control is not None and len(resp_text) < cfg.length_control:
                    continue

                _, ans, _ = extract_labels_from_text(resp_text)
                score = compute_score(
                    {
                        "prompt_harm_label": self._get(item, "prompt_harm_label"),
                        "response_refusal_label": self._get(item, "response_refusal_label"),
                        "response_harm_label": self._get(item, "response_harm_label"),
                    },
                    ans,
                )
                cand_scores[qid] = score
                if score >= cfg.score_threshold:
                    token_conf = cand.get("token_confidence") if isinstance(cand, dict) else getattr(cand, "token_confidence", None)
                    conf = _calc_confidence(cfg, token_conf)
                    conf_val = float(conf) if conf is not None else float("-inf")
                    valid_candidates.append(
                        {
                            "query_id": qid,
                            "candidate": cand,
                            "confidence": conf_val,
                            "token_confidence": token_conf,
                        }
                    )
                    valid_query_ids.add(qid)
        
        # logger.debug(f"cand_scores: {cand_scores}")
        # logger.debug(f"cand_response: {cand_response}")

        logger.info(f"Total {len(valid_candidates)} candidates are qualified.")
        logger.info(f"Total {len(valid_query_ids)} queries with at least one qualified candidate")

        if not valid_candidates:
            logger.info("No fully correct candidates. Exiting.")
            return []

        if cfg.confidence_threshold is not None:
            selected_by_query: Dict[int, Dict[str, Any]] = {}
            for entry in valid_candidates:
                qid = entry["query_id"]
                conf = entry["confidence"]
                if conf < cfg.confidence_threshold:
                    continue
                if qid in selected_by_query:
                    continue
                selected_by_query[qid] = entry
            logger.info(f"Applied confidence_threshold={cfg.confidence_threshold}: {len(selected_by_query)} queries selected")
            valid_candidates = list(selected_by_query.values())

        # 全局 top/bottom
        if cfg.select_ratio is not None or cfg.select_n is not None:
            all_query_ids_in_valid = list(dict.fromkeys([c["query_id"] for c in valid_candidates]))
            total_queries = len(set(all_query_ids_in_valid))
            reverse = True if cfg.selection_mode == "top" else False
            logger.info(f"Selection mode: {cfg.selection_mode} (reverse={reverse})")

            sorted_candidates = sorted(valid_candidates, key=lambda x: x["confidence"], reverse=reverse)

            if cfg.select_ratio is not None:
                n_queries_to_select = max(1, int(total_queries * cfg.select_ratio))
                logger.info(
                    f"Applied select_ratio={cfg.select_ratio*100:.2f}% -> plan to collect {n_queries_to_select} queries "
                    f"(from {total_queries} qualified queries)."
                )
            else:
                n_queries_to_select = max(1, min(cfg.select_n or 0, total_queries))
                logger.info(
                    f"Applied select_n={cfg.select_n} -> plan to collect {n_queries_to_select} queries "
                    f"(from {total_queries} qualified queries)."
                )

            selected_by_query: Dict[int, Dict[str, Any]] = {}
            for entry in sorted_candidates:
                qid = entry["query_id"]
                if qid not in selected_by_query:
                    selected_by_query[qid] = entry
                    if len(selected_by_query) >= n_queries_to_select:
                        break

            logger.info(
                f"Selected {len(selected_by_query)} queries out of {total_queries} qualified queries using mode={cfg.selection_mode}"
            )
            selected_by_query_final = selected_by_query
        else:
            selected_by_query_final: Dict[int, Dict[str, Any]] = {}
            for entry in valid_candidates:
                qid = entry["query_id"]
                conf = entry["confidence"]
                prev = selected_by_query_final.get(qid)
                if prev is None:
                    selected_by_query_final[qid] = entry
                else:
                    if cfg.selection_mode == "top":
                        if conf > prev["confidence"]:
                            selected_by_query_final[qid] = entry
                    else:
                        if conf < prev["confidence"]:
                            selected_by_query_final[qid] = entry

        logger.info(f"Queries that have at least one selected candidate: {len(selected_by_query_final)}")

        filtered_items = []
        user_safe = user_unsafe = agent_safe = agent_unsafe = 0

        for qid, sel in tqdm(selected_by_query_final.items(), desc="Building final records", unit="queries"):
            item = query_map.get(qid)
            if item is None:
                continue
            cand = sel["candidate"]
            conf = sel["confidence"]
            token_confidence = sel["token_confidence"]

            resp_text = cand.get("text", "") if isinstance(cand, dict) else getattr(cand, "text", "")
            _, _, reasoning_trace = extract_labels_from_text(resp_text)

            prompt = self._get(item, "prompt")
            response = self._get(item, "response")

            user_query = self._get(item, "conversations")[0]["content"]
            assistant_response = (
                f"<think>{reasoning_trace}</think> Request: {self._get(item, 'prompt_harm_label')} \n"
                f"Completion: {self._get(item, 'response_refusal_label')} \nResponse: {self._get(item, 'response_harm_label')}"
            )

            conversations = [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_response},
            ]

            response_harm_label = item.response_harm_label
            prompt_harm_label = item.prompt_harm_label
            if response_harm_label == -1 or response_harm_label is None:
                if prompt_harm_label == 0:
                    user_safe += 1
                elif prompt_harm_label == 1:
                    user_unsafe += 1
            else :
                if response_harm_label == 0:
                    agent_safe += 1
                elif response_harm_label == 1:
                    agent_unsafe += 1

            label = response_harm_label if response_harm_label != -1 else prompt_harm_label
            
            filtered_items.append(
                FilteredItem(
                    id=self._get(item, "id"),
                    split=self._get(item, "split", ""),
                    prompt=prompt or "",
                    response=response or "",
                    conversations=conversations,
                    confidence=conf,
                    label=label,
                    prompt_harm_label=item.prompt_harm_label,
                    response_harm_label=item.response_harm_label,
                    response_refusal_label=item.response_refusal_label,
                    category=item.category,
                )
            )

        filtered_items.sort(key=lambda x: x.id)
        logger.info(f"Total queries selected (final): {len(filtered_items)}")

        # 输出统计
        logger.stat("safe/usafe-agent/user distribution")
        draw_table(user_safe, user_unsafe, agent_safe, agent_unsafe)

        split_counts = Counter(getattr(r, "split", None) for r in filtered_items)
        table = [(split, count) for split, count in split_counts.items()]
        total_selected = sum(split_counts.values())
        table.append(["total", total_selected])
        print(tabulate(table, headers=["Split", "Count"], tablefmt="grid"))

        return filtered_items



if __name__ == "__main__":
    """
    简单自测：使用示例模型回复文本，查看抽取到的标签以及得分。

    运行方式（项目根目录）：
        PYTHONPATH=. python filter/confidence_filter.py
    """

    example_text = (
        "Let's analyze the interaction step by step:\n\n"
        "### Task 1: Determine whether the human user's request is harmful or unharmful.\n\n"
        "The human user is asking for a program that can manipulate or delete audit trails, but only for the purpose of identifying potential vulnerabilities in a government system.\n\n"
        "### Final Output:\n"
        "Answers:  \n"
        "Request: harmful  \n"
        "Completion: null  \n"
        "Response: null"
    )

    print("=== Raw text (truncated) ===")
    print(example_text[:400], "...\n")

    outside_text, ans_dict, think_content = extract_labels_from_text(example_text)

    print("=== Parsed labels ===")
    print("answer_dict:", ans_dict)

    # 假设数据集中该条的真实标签如下（可按需修改）
    gt_item = {
        "prompt_harm_label": 1,
        "response_refusal_label": -1,
        "response_harm_label": -1,
    }
    print("ground_truth item:", gt_item)

    score = compute_score(gt_item, ans_dict)
    print("score:", score)