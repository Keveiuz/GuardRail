import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from typing import List, Dict
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import os

from generator.base import BaseGenerator
from utils import GenerationResult, GenerationList, GenerationCandidate

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

@dataclass
class LLMConfig():
    model: str = "/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-8B"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    tensor_parallel_size: int = 1  
    max_logprobs: int = 25
    dtype: str = "float16"

@dataclass
class SamplingConfig():
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    logprobs: int = 25
    prompt_logprobs: int = 0
    n: int = 8

@dataclass
class RayConfig():
    num_actors: int = 1
    num_gpus: int = 2

@dataclass
class DataConfig():
    input_file: str = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-prepare.jsonl"
    output_file: str = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/guardreasoner-metric.jsonl"
    temp_path: str = "/workspace/mnt/lxb_work/zez_work/Boundary-Sample/temp/sft-model-10K"
    num_queries: int = None
    random_sample: bool = True
    batch_processing_size: int = 10000


# 使用Actor模式预加载模型
@ray.remote(num_gpus=LLMConfig.tensor_parallel_size, num_cpus=4)
class LogprobInferenceActor:
    def __init__(self):
        
        llm_config = LLMConfig()
        sampling_config = SamplingConfig()
        
        self.llm = LLM(
            model=llm_config.model,
            tensor_parallel_size=llm_config.tensor_parallel_size,
            gpu_memory_utilization=llm_config.gpu_memory_utilization,
            max_model_len=llm_config.max_model_len,
            dtype=llm_config.dtype,
            trust_remote_code=True,
            enforce_eager=True,
            max_logprobs=llm_config.max_logprobs,
        )
        
        self.sampling_params = SamplingParams(
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            max_tokens=sampling_config.max_tokens,
            logprobs=sampling_config.logprobs,
            n=sampling_config.n,
        )
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """批量生成文本"""

        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for output in outputs:
            prompt_candidates = []
            for resp in output.outputs:
                candidate_dict = {"text": resp.text}
                if hasattr(resp, "logprobs") and resp.logprobs is not None:
                    token_confidences = []
                    for pos_logprobs in resp.logprobs:
                        if pos_logprobs:
                            token_confidence = - sum([pos_logprob.logprob for pos_logprob in pos_logprobs.values()]) / len(pos_logprobs)
                        else:
                            token_confidence = None  # 如果该位置没有 logprobs
                        token_confidences.append(token_confidence)
                    candidate_dict["token_confidence"] = token_confidences
                else:
                    candidate_dict["token_confidence"] = None
                prompt_candidates.append(candidate_dict)
            results.append(prompt_candidates)

        return results


def distribute_prompts(prompts: List[str], num_models: int) -> List[List[str]]:
    """将prompts平均分配给模型"""
    distributed = [[] for _ in range(num_models)]
    for i, prompt in enumerate(prompts):
        model_index = i % num_models
        distributed[model_index].append(prompt)
    return distributed


class LocalInferenceGenerator(BaseGenerator):
    """
    使用 vLLM + Ray 的本地推理引擎
    复用原脚本的配置与生成流程，不改变推理逻辑
    """

    def __init__(self):
        if LLMConfig.tensor_parallel_size * RayConfig.num_actors != RayConfig.num_gpus:
            raise ValueError(
                f"Tensor parallel size * num_actors must equal num_gpus, "
                f"got {LLMConfig.tensor_parallel_size} * {RayConfig.num_actors} != {RayConfig.num_gpus}"
            )

        ray.init(num_gpus=RayConfig.num_gpus)

        # 创建 actor 池（复用原脚本逻辑）
        self.models = []
        for i in range(RayConfig.num_actors):
            model = LogprobInferenceActor.options(name=f"logprob_actor_{i+1}").remote()
            self.models.append(model)

        # tokenizer 仅用于保持原 prompt 构造方式，如需 chat 模板可在 generate 中扩展
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLMConfig.model, trust_remote_code=True
        )

    def generate(self, queries: List[str]) -> GenerationList:
        """
        输入: queries 列表（每个元素为原始 prompt 文本）
        输出: GenerationResult 列表，顺序与输入一致
        """
        prompts = list(queries)
        # 分配到各 actor
        distributed_prompts = distribute_prompts(prompts, RayConfig.num_actors)

        tasks = []
        for model, batch_prompts in zip(self.models, distributed_prompts):
            if batch_prompts:
                task = model.generate_batch.remote(batch_prompts)
                tasks.append(task)
            else:
                tasks.append(None)

        all_results = []
        for task in tasks:
            if task is not None:
                part_results = ray.get(task)
                all_results.extend(part_results)

        # 重排顺序回到原输入顺序
        prompt_to_result = {}
        result_index = 0
        for i, batch_prompts in enumerate(distributed_prompts):
            for j in range(len(batch_prompts)):
                if result_index < len(all_results):
                    original_index = i + j * RayConfig.num_actors
                    if original_index < len(prompts):
                        prompt_to_result[original_index] = all_results[result_index]
                        result_index += 1

        final_results = []
        for i, q in enumerate(prompts):
            candidates_result: List[GenerationCandidate] = []
            candidates = prompt_to_result.get(i, [])
            for j, candidate in enumerate(candidates):
                candidates_result.append(
                    GenerationCandidate(
                        text=candidate["text"],
                        token_confidence=candidate["token_confidence"],
                        token_logprobs=None
                    )
                )

            final_results.append(
                GenerationResult(
                    query=q,
                    candidates=candidates_result,
                    extra={"candidates": candidates},
                )
            )
        return final_results
