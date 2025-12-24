from openai import OpenAI
from typing import List, Tuple, Union

from generator.base import BaseGenerator

from tqdm import tqdm


class AliyunGenerator(BaseGenerator):
    """
    使用阿里云 DashScope 兼容 OpenAI 协议的推理引擎。

    - `generate`: 简单文本生成，返回 List[str]（每个元素是一条完整回复）。
    - `generate_with_thinking`: 使用深度思考模型，返回 (reasoning_content, answer_content)。
    """

    def __init__(
        self,
        api_key: str = "ALIYUN_TOKEN",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3-8b",
        system_prompt: str = "You are a helpful assistant.",
        enable_thinking: bool = False,
        logprobs: bool = False,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.logprobs = logprobs

    def generate(
        self, 
        queries: List[str], 
        model: str = None,
        logprobs: bool = False,
        **kwargs
    ) -> Union[List[dict], List[str]]:
        """
        简单多轮生成：输入多个 query，返回同样长度的回复列表（不包含思考过程）。
        """
        logprobs = logprobs or self.logprobs

        results: List[dict] = []
        responses: list[str] = []

        if logprobs:
            token_confidences: List[float] = []

        for q in tqdm(queries, desc="Generating", unit="query"):
            completion = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": q},
                ] if isinstance(q, str) else q,
                extra_body={
                    "enable_thinking": self.enable_thinking,
                    "logprobs": logprobs,
                    "top_logprobs": 5,
                },
                **kwargs,
            )

            # 获取 logprobs 信息
            if logprobs:
                token_confidence = []
                logprobs_content = completion.choices[0].message.logprobs["content"]
                for k, token_info in enumerate(logprobs_content):
                    # print(f"\nPosition {k}:")
                    # print("Chosen Token:", token_info["token"])
                    # print("Chosen Logprob:", token_info["logprob"])
                    top_candidates = token_info["top_logprobs"]
                    top_logprobs = []
                    for i, top_cand in enumerate(top_candidates):
                        # print(f"    Top candidate {i}:", top_cand["token"])
                        if "logprob" in top_cand.keys():
                            top_logprobs.append(top_cand["logprob"])
                            # print(f"    Top candidate {i} logprob:", top_cand["logprob"])
                        
                    token_confidence.append(- sum(top_logprobs) / len(top_logprobs))
                token_confidences.append(token_confidence)
                
            msg = completion.choices[0].message

            responses.append(msg.content if msg else "")
            
            if logprobs:
                results.append({
                    "response": msg.content if msg else "",
                    "token_confidence": token_confidence
                })

        if logprobs:
            return results

        return responses

    def generate_with_thinking(
        self,
        queries: List[str],
        model: str = None,
    ) -> List[Tuple[str, str]]:
        """
        使用阿里云深度思考模型，返回每个 query 对应的「思考过程」和「最终回复」。

        Args:
            queries: 用户输入内容列表。
            model: 可选，覆盖默认模型（例如 "qwen-plus" / "qwen-long" 等支持 thinking 的模型）。

        Returns:
            List[Tuple[reasoning_content, answer_content]]
        """
        use_model = model or self.model
        results: List[Tuple[str, str]] = []

        for q in queries:
            messages = [{"role": "user", "content": q}]

            completion = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                extra_body={"enable_thinking": True},
                stream=True,
                stream_options={"include_usage": True},
            )

            reasoning_content = ""
            answer_content = ""
            is_answering = False

            for chunk in completion:
                # usage 在无 choices 的 chunk 里返回，这里不强制使用
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 汇总思考内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if not is_answering:
                        # 仅在需要时调试打印，可按需注释
                        # print(delta.reasoning_content, end="", flush=True)
                        pass
                    reasoning_content += delta.reasoning_content

                # 汇总正式回复
                if hasattr(delta, "content") and delta.content:
                    is_answering = True
                    # 同上，只在调试时打印
                    # print(delta.content, end="", flush=True)
                    answer_content += delta.content

            results.append((reasoning_content, answer_content))

        return results


if __name__ == "__main__":
    # engine = AliyunGenerator()
    # outs = engine.generate_with_thinking(["为什么大海是蓝色的？", "为什么天空是蓝色的？"], model="qwen3-8b")
    # for thinking, answer in outs:
    #     print("\n====== 思考过程 ======\n")
    #     print(thinking)
    #     print("\n====== 最终回复 ======\n")
    #     print(answer)

    engine = AliyunGenerator()
    outs = engine.generate(["为什么大海是蓝色的？"], model="qwen3-8b")
    for answer in outs:
        print(answer)