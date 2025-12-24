from openai import OpenAI
from typing import List

from generator.base import BaseGenerator
from utils import GenerationResult, GenerationList, GenerationCandidate


class ModelScopeGenerator(BaseGenerator):
    """
    使用 ModelScope OpenAI 兼容接口的推理引擎
    保持原脚本的调用参数
    """

    def __init__(
        self,
        api_key: str = "MS_TOKEN",
        base_url: str = "https://api-inference.modelscope.cn/v1/",
        model: str = "Qwen/Qwen2.5-72B",
        system_prompt: str = "You are a helpful assistant.",
        stream: bool = True,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt
        self.stream = stream

    def generate(self, queries: List[str]) -> GenerationList:
        results: GenerationList = []
        for q in queries:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": q},
            ]
            if self.stream:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                chunks = []
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                content = "".join(chunks)
                finish_reason = "stream"
                usage = None
                token_logprobs = None
                token_conf = None
            else:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                choice = completion.choices[0]
                content = choice.message.content
                finish_reason = choice.finish_reason
                usage = completion.usage
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
                            text=content or "",
                            token_confidence=token_conf,
                            token_logprobs=token_logprobs,
                        )
                    ],
                    extra={
                        "finish_reason": finish_reason,
                        "usage": usage.model_dump() if usage else None,
                        "stream": self.stream,
                    },
                )
            )
        return results


if __name__ == "__main__":
    engine = ModelScopeGenerator()
    out = engine.generate(["用python写一下快排"])
    for item in out:
        print(item.response)