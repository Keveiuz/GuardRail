from openai import OpenAI
from typing import List

from generator.base import BaseGenerator
from utils import GenerationResult, GenerationList, GenerationCandidate


class HuggingfaceGenerator(BaseGenerator):
    """
    使用 HuggingFace Router (OpenAI 兼容) 的推理引擎
    保持原脚本的调用参数
    """

    def __init__(
        self,
        api_key: str = "HF_TOKEN",
        base_url: str = "https://router.huggingface.co/v1",
        model: str = "Qwen/Qwen2.5-72B:featherless-ai",
        system_prompt: str = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

    def generate(self, queries: List[str]) -> GenerationList:
        results: GenerationList = []
        for q in queries:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": q})

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            choice = completion.choices[0]
            msg = choice.message
            token_logprobs = None
            token_conf = None
            if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None):
                token_logprobs = [tp.logprob for tp in choice.logprobs.content if tp is not None]
                if token_logprobs:
                    token_conf = sum(token_logprobs) / len(token_logprobs)
            results.append(
                GenerationResult(
                    query=q,
                    candidates=[
                        GenerationCandidate(
                            text=msg.content if msg else "",
                            token_confidence=token_conf,
                            token_logprobs=token_logprobs,
                        )
                    ],
                    extra={
                        "finish_reason": choice.finish_reason,
                        "usage": completion.usage.model_dump() if getattr(completion, "usage", None) else None,
                        "raw": completion.model_dump(),
                    },
                )
            )
        return results


if __name__ == "__main__":
    engine = HuggingfaceGenerator()
    out = engine.generate(["What is the capital of France?"])
    for item in out:
        print(item.response)