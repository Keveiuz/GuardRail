from datasets import load_dataset
from .base import BaseDatasetLoader
from utils import GuardrailDataset
from utils.logger import Logger
from tqdm import tqdm



class ToxicChatLoader(BaseDatasetLoader):
    """
    Loader for: lmsys/toxic-chat (subset: toxicchat0124, split: train)

    - 该数据集只有一个 toxicity 标签，不区分 user/agent；
      因测试集中 toxicchat 属于 user 任务，因此使用 user_input 作为 prompt。
    - prompt_harm_label 直接由 toxicity 转为 int（0/1）。
    - response_harm_label、response_refusal_label、category 设为 None。
    - category 为 openai_moderation 概率最大的类别；否则 category=None。
    - extra 字段包含原始字段 "jailbreaking"。

    返回结构:
    {
        prompt=user_input,
        response=model_output,
        prompt_harm_label,
        response_harm_label=None,
        response_refusal_label=None,
        category=None,
        extra={"jailbreaking": ...}
    }
    """

    name = "toxicchat"

    def load(self) -> GuardrailDataset:
        logger = Logger()

        logger.info("Loading ToxicChat dataset")
        ds = load_dataset("/Users/jason/Documents/GuardRail/datasets/toxicchat", "toxicchat0124", split="train")
        # ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
        logger.info(f"Loaded {len(ds)} examples")
        result = []

        for item in tqdm(ds, desc="Processing ToxicChat dataset"):
            prompt_harm_label = int(item["toxicity"])

            if prompt_harm_label is None:
                raise ValueError(
                    f"Invalid value for prompt_harm_label: None. "
                )

            category = None
            if prompt_harm_label:
                category = item["openai_moderation"][0][0]

            result.append(self.make_item(
                prompt=item['user_input'],
                response=item['model_output'],
                prompt_harm_label=prompt_harm_label,
                response_harm_label=None,
                response_refusal_label=None,
                category=category,
                extra={"jailbreaking": item["jailbreaking"]}
            ))

        return result
