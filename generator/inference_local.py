import json
from typing import List, Dict
from dataclasses import dataclass

from generator.base import BaseGenerator
from utils import GenerationResult, GenerationList, GenerationCandidate


class LocalInferenceGenerator(BaseGenerator):
    """
    vLLM + Ray 本地推理引擎（纯推理版）
    - 无 resume
    - 无中间文件
    - generate() 直接返回结果
    """

    # ================= Config =================

    @dataclass
    class LLMConfig:
        model: str = "/data/pretrained_models/Qwen2.5/Qwen2.5-72B-GeoGPT"
        gpu_memory_utilization: float = 0.7
        max_model_len: int = 4096
        tensor_parallel_size: int = 2
        max_logprobs: int = 25
        dtype: str = "float16"

    @dataclass
    class SamplingConfig:
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 2048
        logprobs: int = 25
        n: int = 1

    @dataclass
    class RayConfig:
        num_actors: int = 1
        num_gpus: int = 2

    # ================= Init =================

    def __init__(
        self,
        llm_config: LLMConfig,
        sampling_config: SamplingConfig,
        ray_config: RayConfig,
    ):
        self.llm_config = llm_config
        self.sampling_config = sampling_config
        self.ray_config = ray_config

    # ================= Utils =================

    def distribute_prompts(self, prompts: List[str], num_actors: int):
        buckets = [[] for _ in range(num_actors)]
        for i, p in enumerate(prompts):
            buckets[i % num_actors].append((i, p))
        return buckets

    # ================= Core =================

    def generate(self, queries: List[Dict], logprobs: bool = False) -> List[Dict]:

        import ray
        import subprocess
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        # try:
        #     subprocess.run("ray stop", shell=True)
        #     subprocess.run("ps -eaf | grep VLLM::EngineCore | grep -v grep | awk '{print $2}' | xargs -r kill -9", shell=True)
        # except Exception:
        #     pass

        # ---------- Ray Actor ----------

        @ray.remote(num_gpus=self.llm_config.tensor_parallel_size, num_cpus=4)
        class InferenceActor:
            def __init__(self, llm_config, sampling_config):
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

            def generate_batch(self, indexed_prompts):
                indices, prompts = zip(*indexed_prompts)
                outputs = self.llm.generate(list(prompts), self.sampling_params)

                results = []
                for out in outputs:
                    candidates = []
                    for resp in out.outputs:
                        cand = {"text": resp.text}
                        if resp.logprobs is not None:
                            token_conf = []
                            for pos in resp.logprobs:
                                if pos:
                                    token_conf.append(
                                        -sum(lp.logprob for lp in pos.values()) / len(pos)
                                    )
                                else:
                                    token_conf.append(None)
                            cand["token_confidence"] = token_conf
                        else:
                            cand["token_confidence"] = None
                        candidates.append(cand)
                    results.append(candidates)

                return list(zip(indices, results))

            def cleanup(self):
                if hasattr(self, 'llm') and self.llm is not None:
                    # vLLM V1 必须显式调用引擎的 shutdown 才能杀死底层 Worker 进程
                    if hasattr(self.llm.llm_engine, 'shutdown'):
                        self.llm.llm_engine.shutdown()
                    del self.llm
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()

        # ---------- Sanity Check ----------

        assert (
            self.llm_config.tensor_parallel_size * self.ray_config.num_actors
            == self.ray_config.num_gpus
        ), "tensor_parallel_size * num_actors 必须等于 num_gpus"

        # ---------- Init ----------

        ray.init(
            address="10.222.14.191:6360",
            dashboard_port="8269",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_config.model, trust_remote_code=True
        )

        # ---------- Build Prompts ----------

        prompts = []
        for item in queries:
            conv = [{"role": "user", "content": item}] if isinstance(item, str) else item
            prompt = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            
            token_len = len(
                tokenizer.encode(prompt, add_special_tokens=False)
            )

            if token_len > self.llm_config.max_model_len - self.sampling_config.max_tokens:
                prompts.append("too long, skipped")
            else:
                prompts.append(prompt)

        distributed = self.distribute_prompts(
            prompts, self.ray_config.num_actors
        )

        # ---------- Actors ----------

        actors = [
            InferenceActor.remote(self.llm_config, self.sampling_config)
            for _ in range(self.ray_config.num_actors)
        ]

        tasks = []
        for actor, bucket in zip(actors, distributed):
            if bucket:
                tasks.append(actor.generate_batch.remote(bucket))

        # ---------- Gather Results ----------

        final_results = [None] * len(prompts)
        for task in tasks:
            for idx, result in ray.get(task):
                final_results[idx] = result

        # ---------- Attach Back ----------

        # ---------- 【关键修改：彻底释放】 ----------
        # # 1. 先让 Actor 内部执行 vLLM 销毁逻辑
        # ray.get([actor.cleanup.remote() for actor in actors])
        
        # # 2. 杀死 Actor 进程
        # for actor in actors:
        #     ray.kill(actor)
        
        # # 3. 关闭 Ray 环境
        ray.shutdown()
        
        # # 4. 强制主进程回收
        # import gc
        # gc.collect()
        
        return final_results


# =========================================================
# Minimal Test
# =========================================================

if __name__ == "__main__":

    # ⚠️ 推荐在 shell 中设置
    # CUDA_VISIBLE_DEVICES=1,3 python this_file.py

    llm_config = LocalInferenceGenerator.LLMConfig(
        tensor_parallel_size=2,
    )

    sampling_config = LocalInferenceGenerator.SamplingConfig(
        max_tokens=128,
        logprobs=5,
    )

    ray_config = LocalInferenceGenerator.RayConfig(
        num_actors=1,
        num_gpus=2,
    )

    generator = LocalInferenceGenerator(
        llm_config=llm_config,
        sampling_config=sampling_config,
        ray_config=ray_config,
    )

    test_queries = [
        "请用一句话解释什么是重力。"
    ]

    print("\n🚀 Start inference test...\n")

    results = generator.generate(test_queries)

    print("\n✅ Inference finished.\n")

    print(json.dumps(results[0]["candidates"], ensure_ascii=False, indent=2))
    print("\n🎉 纯推理测试通过（无 resume / 无中间文件）")
