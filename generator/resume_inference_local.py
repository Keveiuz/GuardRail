import json
from tqdm import tqdm
from typing import List, Dict, Type, Optional, cast
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import os
import functools

# 假设这些类和模块已存在
from generator.base import BaseGenerator
from utils import GenerationResult, GenerationList, GenerationCandidate

# 保持环境变量设置不变
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"


# --- LocalInferenceGenerator ---

class LocalInferenceGenerator(BaseGenerator):
    """
    使用 vLLM + Ray 的本地推理引擎，通过传入参数配置模型和并行度。
    """

    @dataclass
    class LLMConfig():
        model: str = "/data/pretrained_models/Qwen2.5/Qwen2.5-72B-GeoGPT"
        gpu_memory_utilization: float = 0.95
        max_model_len: int = 4096
        tensor_parallel_size: int = 2
        max_logprobs: int = 25
        dtype: str = "float16"

    @dataclass
    class SamplingConfig():
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 4096
        logprobs: int = 25
        prompt_logprobs: int = 0
        n: int = 1

    @dataclass
    class RayConfig():
        num_actors: int = 1
        num_gpus: int = 2

    @dataclass
    class DataConfig():
        input_file: str = "/data/zez/Boundary/temp/guardreasoner-prepare.jsonl"
        output_file: str = "/data/zez/Boundary/temp/guardreasoner-metric.jsonl"
        temp_path: str = "./temp"
        num_queries: int = None
        random_sample: bool = True
        batch_processing_size: int = 10000

    llm_config = LLMConfig()
    sampling_config = SamplingConfig()
    ray_config = RayConfig()
    data_config = DataConfig

    def __init__(self, llm_config: LLMConfig, sampling_config: SamplingConfig, ray_config: RayConfig, data_config: DataConfig):

        self.llm_config = llm_config
        self.sampling_config = sampling_config
        self.ray_config = ray_config
        self.data_config = data_config


    def distribute_prompts(self, prompts: List[str], num_models: int) -> List[List[str]]:
        """将prompts平均分配给模型"""
        distributed = [[] for _ in range(num_models)]
        for i, prompt in enumerate(prompts):
            model_index = i % num_models
            distributed[model_index].append(prompt)
        return distributed
    
    def load_dataset(self, file_path: str, num_queries: int = None, random_sample: bool = False) -> List[Dict]:
        import random
        random.seed(42)
        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                if num_queries is not None:
                    if random_sample:
                        df = df.sample(n=num_queries, random_state=42)
                    else:
                        df = df.head(num_queries)
                return df.to_dict('records')

            elif file_path.endswith('.jsonl'):
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
                
                if num_queries is not None:
                    if random_sample:
                        data = random.sample(data, min(num_queries, len(data)))
                    else:
                        data = data[:num_queries]
                return data

            else:
                raise ValueError(f"Unsupport File Format: {file_path}")

        except Exception as e:
            raise ValueError(f"Unable to load file: {file_path}, Exception: {str(e)}")

    def save_data(self, data: list, file_path: str):
        """保存数据"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif path.suffix == '.jsonl':
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif path.suffix == '.parquet':
            pd.DataFrame(data).to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")



    def generate(self, queries: List[str], logprobs: bool = False) -> GenerationList:

        import ray
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        from tqdm import tqdm
        from typing import List, Dict
        from pathlib import Path

        # 使用Actor模式预加载模型
        @ray.remote(num_gpus=self.llm_config.tensor_parallel_size, num_cpus=4)
        class LogprobInferenceActor:
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
            
            def generate_batch(self, prompts: List[str]) -> List[str]:
                """批量生成文本"""
                
                if isinstance(prompts[0], str):
                    outputs = self.llm.generate([
                        [
                            {"role": "user", "content": prompt},
                        ] for prompt in prompts
                    ], self.sampling_params)
                else:
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
            

        if self.llm_config.tensor_parallel_size * self.ray_config.num_actors != self.ray_config.num_gpus:
            raise ValueError(
                f"Tensor parallel size * num_actors must equal num_gpus, "
                f"got {self.llm_config.tensor_parallel_size} * {self.ray_config.num_actors} != {self.ray_config.num_gpus}"
            )
        
        # 初始化Ray
        ray.init(num_gpus=self.ray_config.num_gpus)
        
        tokenizer = AutoTokenizer.from_pretrained(self.llm_config.model, trust_remote_code=True)
        messages = queries

        # 创建模型实例
        tqdm.write(f"🧠 Initializing {self.ray_config.num_actors} actors with model {self.llm_config.model} ...")
        models = []
        for i in range(self.ray_config.num_actors):
            model = LogprobInferenceActor.options(name=f"logprob_actor_{i+1}").remote(self.llm_config, self.sampling_config)
            models.append(model)

        # === 准备输出目录 ===
        temp_dir = Path(self.data_config.temp_path)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # === 检查已有的 batch 进度（resume） ===
        existing_parts = sorted(temp_dir.glob("part_*.jsonl"))
        if existing_parts:
            completed_batches = [int(p.stem.split("_")[1]) for p in existing_parts if p.stem.split("_")[1].isdigit()]
            resume_from = max(completed_batches) + 1 if completed_batches else 0
            tqdm.write(f"🔄 Resuming from batch {resume_from} (found {len(existing_parts)} completed parts)")
        else:
            resume_from = 0
            tqdm.write("🚀 Starting from scratch (no existing partial results found)")

        # === 分批处理 ===
        batch_size = self.data_config.batch_processing_size if self.data_config.batch_processing_size is not None else len(messages)
        num_batches = (len(messages) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(resume_from, num_batches),
                            desc="🚀 Processing batches",
                            unit="batch",
                            position=0,       # 外层进度条位置
                            leave=True,
                            dynamic_ncols=True):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(messages))
            batch_messages = messages[start:end]

            tqdm.write(f"📦 Batch {batch_idx + 1}/{num_batches} "
                    f"({start} ~ {end - 1}, total {len(batch_messages)} prompts)")

            prompts = []
            batch_results = []

            for item in tqdm(batch_messages,
                            desc="Tokenize Prompt",
                            position=1,      # 内层进度条位置
                            leave=False,
                            dynamic_ncols=True):
                prompt = tokenizer.apply_chat_template(
                    item["conversations"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )

                batch_results.append(item)
                prompts.append(prompt)

            # 将prompts平均分配给模型
            distributed_prompts = self.distribute_prompts(prompts, self.ray_config.num_actors)
            
            # 为每个模型创建批量推理任务
            tasks = []
            for i, (model, batch_prompts) in enumerate(zip(models, distributed_prompts)):
                print(f"Model {i+1} will process {len(batch_prompts)} prompts")
                if batch_prompts:  # 只有当有prompts时才创建任务
                    task = model.generate_batch.remote(batch_prompts)
                    tasks.append(task)
                else:
                    tasks.append(None)  # 没有prompts的模型
            
            # 批量获取所有结果
            all_results = []
            for i, task in enumerate(tasks):
                if task is not None:
                   all_results.extend(ray.get(task))
            
            # 将结果添加到数据中（保持原始顺序）
            prompt_to_result = {}
            result_index = 0
            
            # 重新分配结果到对应的prompt
            for i, batch_prompts in enumerate(distributed_prompts):
                for j in range(len(batch_prompts)):
                    if result_index < len(all_results):
                        # 找到这个prompt在原始prompts中的位置
                        original_index = i + j * self.ray_config.num_actors
                        if original_index < len(prompts):
                            prompt_to_result[original_index] = all_results[result_index]
                            result_index += 1
            
            # 按照原始顺序整理结果
            final_results = []
            for i in range(len(prompts)):
                final_results.append(prompt_to_result.get(i, "Error: Result not found"))
            
            # 将结果添加到数据中
            for item, result in zip(batch_results, final_results):
                item['candidates'] = result

            # 保存临时文件
            part_file = temp_dir / f"part_{batch_idx:03d}.jsonl"
            tqdm.write(f"💾 Saving partial results to {part_file}")
            self.save_data(batch_results, part_file)

    
        # 按照 part_*.jsonl 的顺序读取并合并数据
        print(f"\n🔗 Merging all parts into a Python list from {temp_dir}") 
        
        merged_results: List[Dict] = []
        for part_file in sorted(temp_dir.glob("part_*.jsonl")):
            try:
                with open(part_file, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        # 每一行是一个 JSON 对象，即一个 item
                        merged_results.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ Warning: Failed to read or parse file {part_file}: {e}")
                
        print(f"✅ All batches processed and merged successfully! Total items: {len(merged_results)}")

        ray.shutdown()

        return merged_results




if __name__ == "__main__":
    """
    最小推理测试：
    - 1 条数据
    - 1 个 batch
    - 验证 vLLM + Ray 能否返回 candidates
    """

    # ⚠️ 强烈建议：真正运行时在 shell 里设
    # CUDA_VISIBLE_DEVICES=1,3 python this_file.py

    # ========= 1. 构造配置 =========
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

    llm_config = LocalInferenceGenerator.LLMConfig(
        model="/data/pretrained_models/Qwen2.5/Qwen2.5-72B-GeoGPT",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        dtype="float16",
    )

    sampling_config = LocalInferenceGenerator.SamplingConfig(
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,   # ✅ 测试时一定要小
        n=1,
        logprobs=5,
    )

    ray_config = LocalInferenceGenerator.RayConfig(
        num_actors=1,
        num_gpus=2,
    )

    data_config = LocalInferenceGenerator.DataConfig(
        temp_path="./temp",
        batch_processing_size=10,
    )

    # ========= 2. 初始化 Generator =========
    generator = LocalInferenceGenerator(
        llm_config=llm_config,
        sampling_config=sampling_config,
        ray_config=ray_config,
        data_config=data_config,
    )

    # ========= 3. 构造最小测试数据 =========
    test_queries = [
        {
            "id": "test_001",
            "conversations": [
                {
                    "role": "user",
                    "content": "请用一句话解释什么是重力。"
                }
            ]
        }
    ]

    # ========= 4. 执行推理 =========
    print("\n🚀 Start inference test...\n")

    results = generator.generate(test_queries)

    # ========= 5. 验证结果 =========
    print("\n✅ Inference finished.\n")

    assert results is not None, "generate() 返回了 None"
    assert isinstance(results, list), "返回结果不是 list"
    assert len(results) == 1, "返回条数不正确"

    item = results[0]
    assert "candidates" in item, "结果中缺少 candidates"

    print("🧪 Sample output (candidates):")
    print(json.dumps(item["candidates"], ensure_ascii=False, indent=2))

    print("\n🎉 Test passed：vLLM + Ray 推理链路已跑通！")
