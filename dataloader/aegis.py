from typing import Any


from datasets import load_dataset
from tqdm import tqdm

from .base import BaseDatasetLoader
from utils import GuardrailDataset
from utils.logger import Logger

class AegisV1Loader(BaseDatasetLoader):
    """
    Loader for: nvidia/Aegis-AI-Content-Safety-Dataset-1.0 (train)

    - 仅使用 text_type == "user_message" 的数据。
    - category 由 labels_0~4 的多数投票决定（并列最多 → None）。
    - prompt_harm_label 根据 labels_0/1/2 中是否存在非 Safe 判定：
        Safe → 0, 否则 → 1。
    - response_harm_label 与 response_refusal_label 均设为 None（无可靠数据）。

    返回结构:
    {
        prompt, response,
        prompt_harm_label,
        response_harm_label=None,
        response_refusal_label=None,
        category,
        extra=None
    }
    """
    name = "aegis_v1"

    def load(self) -> GuardrailDataset:
        logger = Logger()

        logger.info("Loading Aegis V1 dataset")
        ds = load_dataset("/data1/zez/OpenSafeGuard/datasets/aegis-v1", split="train")
        # ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", split="train")
        logger.info(f"Loaded {len(ds)} examples")

        def majority_vote(labels):
            from collections import Counter
            if not labels:
                return None
            counter = Counter(labels)
            most_common = counter.most_common()
            top_label, top_count = most_common[0]

            return top_label


        result = []
        for item in tqdm(ds, desc="Processing Aegis V1 dataset"):
            
            # if item["text_type"] == "user_message":
            if item["text_type"] is not None:
                
                labels = [item.get(f"labels_{i}") for i in range(5) if item.get(f"labels_{i}") is not None]
            
                category = majority_vote(labels=labels)

                is_harmful = all(label != "Safe" for label in labels)

                if is_harmful:
                    prompt_harm_label = 1
                else:
                    prompt_harm_label = 0
                
                result.append(self.make_item(
                    prompt=item["text"],
                    response=None,
                    prompt_harm_label=prompt_harm_label,
                    response_harm_label=None,
                    response_refusal_label=None,
                    category=category,
                    extra={
                        "text_type": item.get("text_type"),
                        "labels": [item.get(f"labels_{i}") for i in range(5)]
                    }
                ))

        return result



class AegisV2Loader(BaseDatasetLoader):
    """
    Loader for: nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (train)

    - prompt_label: "safe"→0, "unsafe"→1（否则报错）。
    - response_label: "safe"→0, "unsafe"→1（未知值→None）。
    - response_refusal_label 恒为 None。
    - category 使用 violated_categories 字段。

    返回结构:
    {
        prompt, response,
        prompt_harm_label,
        response_harm_label,
        response_refusal_label=None,
        category,
        extra=None
    }
    """

    name = "aegis_v2"

    def load(self) -> GuardrailDataset:
        logger = Logger()

        logger.info("Loading Aegis V2 dataset")
        ds = load_dataset("/Users/jason/Documents/GuardRail/datasets/aegis_v2", split="train")
        # ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
        logger.info(f"Loaded {len(ds)} examples")
        result = []

        for item in tqdm(ds, desc="Processing Aegis V2 dataset"):
            prompt_harm_label = None 
            if item["prompt_label"] == "safe":
                prompt_harm_label = 0
            elif item["prompt_label"] == "unsafe":
                prompt_harm_label = 1

            if prompt_harm_label is None:
                raise ValueError(
                    f"Invalid value for prompt_harm_label: None. "
                    f"Expected 'safe' or 'unsafe'."
                )
            
            response_harm_label = None 
            if item["response_label"] == "safe":
                response_harm_label = 0
            elif item["response_label"] == "unsafe":
                response_harm_label = 1

            result.append(self.make_item(
                prompt=item["prompt"],
                response=item["response"],
                prompt_harm_label=prompt_harm_label,
                response_harm_label=response_harm_label,
                response_refusal_label=None,
                category=item["violated_categories"],
                extra=None
            ))

        return result
