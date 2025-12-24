from datasets import load_dataset
from .base import BaseDatasetLoader
from utils import GuardrailDataset
from utils.logger import Logger
from tqdm import tqdm


class WildGuardLoader(BaseDatasetLoader):
    """
    Loader for: allenai/wildguardmix (subset: wildguardtrain, split: wildguardtrain)

    - prompt_harm_label: "unharmful"→0, "harmful"→1（否则报错）。
    - response_harm_label: 同上规则，未知值→None。
    - response_refusal_label: "compliance"→0, "refusal"→1，未知值→None。
    - category 使用字段 subcategory。
    - extra 包含 adversarial 字段。

    返回结构:
    {
        prompt, response,
        prompt_harm_label,
        response_harm_label,
        response_refusal_label,
        category,
        extra={"adversarial": ...}
    }
    """

    name = "wildguard"
    
    def load(self) -> GuardrailDataset:
        logger = Logger()

        logger.info("Loading WildGuard dataset")
        ds = load_dataset("/workspace/mnt/lxb_work/zez_work/GuardRail/dataset/wildguard", "wildguardtrain", split="train")
        # ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
        logger.info(f"Loaded {len(ds)} examples")
        result = []

        for item in tqdm(ds, desc="Processing WildGuard dataset"):
            prompt_harm_label = None 
            if item["prompt_harm_label"] == "unharmful":
                prompt_harm_label = 0
            elif item["prompt_harm_label"] == "harmful":
                prompt_harm_label = 1

            if prompt_harm_label is None:
                raise ValueError(
                    f"Invalid value for prompt_harm_label: None. "
                    f"Expected 'unharmful' or 'harmful'."
                )
            
            response_harm_label = None 
            if item["response_harm_label"] == "unharmful":
                response_harm_label = 0
            elif item["response_harm_label"] == "harmful":
                response_harm_label = 1
            
            response_refusal_label = None 
            if item["response_refusal_label"] == "compliance":
                response_refusal_label = 0
            elif item["response_refusal_label"] == "refusal":
                response_refusal_label = 1

            result.append(self.make_item(
                prompt=item["prompt"],
                response=item["response"],
                prompt_harm_label=prompt_harm_label,
                response_harm_label=response_harm_label,
                response_refusal_label=response_refusal_label,
                category=item["subcategory"],
                extra={"adversarial": item["adversarial"]}
            ))

        return result
