from typing import List, Optional

import numpy as np


class ConfidenceCalculator:
    """
    计算论文中提到的多种 confidence 指标。
    输入为每个 token 位置的 confidence 值列表。
    """

    def __init__(self, token_confidences: List[float]):
        """
        Args:
            token_confidences: 每个 token 位置的 confidence 值 [C_0, C_1, ..., C_N]
        """
        if token_confidences is None or len(token_confidences) == 0:
            raise ValueError("token_confidences 不能为空")

        self.conf_list = np.asarray(token_confidences, dtype=float)
        self.n_tokens = self.conf_list.shape[0]

    # ========== 2. Average Trace Confidence ==========
    def compute_average_trace_confidence(self) -> float:
        return float(np.mean(self.conf_list))

    # ========== 3. Group Confidence ==========
    def compute_group_confidence(self, window_size: int = 2048) -> float:
        group_conf = self.compute_group_confidence_positional(window_size=window_size)
        return float(np.mean(group_conf))

    def compute_group_confidence_positional(self, window_size: int = 2048) -> np.ndarray:
        conf = self.conf_list
        n = self.n_tokens
        w = max(1, min(window_size, n))

        prefix = np.zeros(n + 1, dtype=float)
        prefix[1:] = np.cumsum(conf)

        idx = np.arange(n)
        left = np.maximum(idx - w + 1, 0)
        lengths = idx - left + 1
        window_sums = prefix[idx + 1] - prefix[left]
        group_conf = window_sums / lengths
        return group_conf

    # ========== 4. Bottom X% Group Confidence ==========
    def compute_bottom_percent_group_confidence(
        self, window_size: int = 2048, bottom_percent: float = 0.1
    ) -> float:
        group_conf = self.compute_group_confidence_positional(window_size=window_size)
        w = max(1, min(window_size, self.n_tokens))
        valid_group_conf = group_conf[w - 1 :]
        if len(valid_group_conf) == 0:
            return float(np.mean(self.conf_list))

        percent = bottom_percent
        if percent > 1:
            percent = percent / 100.0
        percent = min(max(percent, 0.0), 1.0)

        k = max(1, int(len(valid_group_conf) * percent))
        bottom_k_conf = np.partition(valid_group_conf, k - 1)[:k]
        return float(np.mean(bottom_k_conf))

    # ========== 5. Lowest Group Confidence ==========
    def compute_lowest_group_confidence(self, window_size: int = 2048) -> float:
        group_conf = self.compute_group_confidence_positional(window_size)
        w = max(1, min(window_size, self.n_tokens))
        valid_group_conf = group_conf[w - 1 :]
        if len(valid_group_conf) == 0:
            return float(np.min(self.conf_list))
        return float(np.min(valid_group_conf))

    # ========== 6. Tail Confidence ==========
    def compute_tail_confidence(self, tail_size: int = 2048) -> float:
        size = max(1, min(tail_size, self.n_tokens))
        tail_tokens = self.conf_list[-size:]
        return float(np.mean(tail_tokens))

    def compute_tail_confidence_by_percent(self, tail_percent: float = 0.1) -> float:
        percent = tail_percent
        if percent > 1:
            percent = percent / 100.0
        percent = min(max(percent, 0.0), 1.0)
        size = max(1, int(self.n_tokens * percent))
        tail_tokens = self.conf_list[-size:]
        return float(np.mean(tail_tokens))

    # ========== 综合报告 ==========
    def compute_all_metrics(
        self,
        window_size: int = 2048,
        bottom_percent: float = 0.1,
        tail_size: int = 2048,
        tail_percent: float = 0.1,
    ) -> dict:
        metrics = {
            "average_trace_conf": self.compute_average_trace_confidence(),
            "group_conf_mean": self.compute_group_confidence(window_size=window_size),
            "bottom_group_conf": self.compute_bottom_percent_group_confidence(
                window_size=window_size, bottom_percent=bottom_percent
            ),
            "lowest_group_conf": self.compute_lowest_group_confidence(window_size=window_size),
            "tail_conf_fixed": self.compute_tail_confidence(tail_size),
            "tail_conf_percent": self.compute_tail_confidence_by_percent(tail_percent),
            "n_tokens": self.n_tokens,
        }
        metrics["group_conf_array"] = self.compute_group_confidence_positional(window_size)
        return metrics

