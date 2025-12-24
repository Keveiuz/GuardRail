from datasets import load_dataset
from .base import BaseDatasetLoader
from tqdm import tqdm
from utils import GuardrailDataset
from utils.logger import Logger


class BeavorTailsLoader(BaseDatasetLoader):
    """
    Loader for: PKU-Alignment/BeaverTails (subset: 30k_train)

    - response_harm_label: 0 if is_safe=True, else 1。
    - category 由 categories 字典解析：
        · 若仅一个类别为 True → 返回该类别名；
        · 若全部为 False → None；
        · 若多个为 True → 抛异常。
    - prompt_harm_label、response_refusal_label 设为 None（数据未提供）。
    - extra 设为 None。

    返回结构:
    {
        prompt,
        response,
        prompt_harm_label,
        response_harm_label,
        response_refusal_label,
        category,
        extra
    }
    """

    name = "beavortails"

    def load(self) -> GuardrailDataset:
        logger = Logger()

        logger.info("Loading BeaverTails dataset")
        ds = load_dataset("/Users/jason/Documents/GuardRail/datasets/beavertails", split="30k_train")
        # ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
        logger.info(f"Loaded {len(ds)} examples")
        
        result = []

        def extract_category(category_dict: dict):
            true_keys = [k for k, v in category_dict.items() if v is not False]

            if len(true_keys) == 0:
                return None
            if len(true_keys) == 1:
                return true_keys[0]

            raise ValueError(f"Multiple categories are True: {true_keys}")

        for item in tqdm(ds, desc="Processing BeaverTails dataset"):
            response_harm_label = 0 if item["is_safe"] else 1
            category = extract_category(item["category"])

            result.append(self.make_item(
                prompt=item["prompt"],
                response=item["response"],
                prompt_harm_label=None,
                response_harm_label=response_harm_label,
                response_refusal_label=None,
                category=None,
                extra=None
            ))

        return result
