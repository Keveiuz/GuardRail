import random
import json
from dataclasses import asdict

from dataloader import AegisV1Loader
from generator import LocalInferenceGenerator
from formator import AegisV1SplitFormator
from parser import AegisV1SplitParser

from utils import GenerationCandidate, GenerationResult, GenerationList, GuardrailItem, GuardrailDataset
from utils import Logger
logger = Logger()

aegis_loader = AegisV1Loader()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

llm_config = LocalInferenceGenerator.LLMConfig(
    model="/data/pretrained_models/Qwen3/Qwen3-32B",
    tensor_parallel_size=1,
    max_logprobs=0,
)
sampling_config = LocalInferenceGenerator.SamplingConfig(logprobs=None)
ray_config = LocalInferenceGenerator.RayConfig(num_actors=1, num_gpus=1)
base_engine = LocalInferenceGenerator(
    llm_config=llm_config,
    sampling_config=sampling_config,
    ray_config=ray_config,
)


formator = AegisV1SplitFormator()
parser = AegisV1SplitParser()


MAX_RETRIES = 3  # 最大重试次数

# for loader in [aegis_v1_loader, aegis_v2_loader, beavortails_loader, toxicchat_loader, wildguard_loader]:
for loader in [aegis_loader]:

    logger.info(f"Loading {loader.name} dataset")
    data = loader.load()
    logger.info(f"{loader.name} dataset loaded, size={len(data)}")

    logger.info(f"Formatting {loader.name} dataset")
    queries = formator.format(data=data, split=loader.name)
    logger.info(f"{loader.name} formatted, queries={len(queries)}")

    # 初始批量生成
    logger.info(f"Generating {loader.name} dataset")
    inference_result = base_engine.generate(queries=[query.conversations for query in queries])
    logger.info(f"{loader.name} generation done, results={len(inference_result)}")

    # 批量解析
    logger.info(f"Parsing results")
    result = parser.parse([infer[0]['text'] for infer in inference_result])
    logger.info(f"parsing done")

    # ============================
    # retry logic start
    # ============================
    retries = 0
    failed_indices = [
        i for i, infer in enumerate(result)
        if (infer.get('query') is None and infer.get('response') is None)
    ]

    while failed_indices and retries < MAX_RETRIES:
        retries += 1
        logger.warn(f"{len(failed_indices)} items failed parsing, retry {retries}/{MAX_RETRIES}")

        # 取失败条目的 conversations 批量生成
        retry_queries = [queries[i].conversations for i in failed_indices]
        retry_result_raw = base_engine.generate(queries=retry_queries)
        retry_parsed = parser.parse([infer[0]['text'] for infer in retry_result_raw])

        # 替换原 result 中的失败条目
        for idx_in_list, idx_original in enumerate(failed_indices):
            result[idx_original] = retry_parsed[idx_in_list]

        # 更新下一轮失败条目
        failed_indices = [
            i for i, infer in enumerate(result)
            if (infer.get('query') is None and infer.get('response') is None)
        ]
    # ============================
    # retry logic end
    # ============================

    # 生成最终结果
    generation_result: GenerationList = []
    for n, (raw, infer) in enumerate(zip(queries, result)):
        generation_result.append(
            GenerationResult(
                id=f"aegis-v1-{n:06d}",
                split=raw.split,
                prompt=infer['query'] if isinstance(infer['query'], str) else None,
                response=infer['response'] if isinstance(infer['response'], str) else None,
                conversations=raw.conversations,
                prompt_harm_label=raw.prompt_harm_label,
                response_harm_label=raw.response_harm_label,
                response_refusal_label=raw.response_refusal_label,
                category=raw.category,
                extra={
                    **raw.extra,
                    'split_query': infer['query'],
                    'split_response': infer['response'],
                    'raw': raw.prompt
                },
                candidates=[],
            )
        )

    # 将 GuardrailItem 列表序列化为 JSONL
    logger.info(f"Saving results to synth_data_{loader.name}.jsonl")
    with open(f"synth_data_{loader.name}.jsonl", "w", encoding="utf-8") as f:
        for gi in generation_result:
            f.write(json.dumps(asdict(gi), ensure_ascii=False) + "\n")
    logger.info(f"Result saved")
