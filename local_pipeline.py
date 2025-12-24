import random
import json
import re

from dataclasses import asdict

from dataloader import WildGuardLoader
from generator import LocalInferenceGenerator
from formator import WildGuardFormator
from parser import WildGuardParser
from filter import ConfidenceFilter
from synthesizer import SelfInstruct

from utils import GenerationCandidate, GenerationResult, GenerationList, GuardrailItem, GuardrailDataset
from utils import Logger
logger = Logger()

wildguard_loader = WildGuardLoader()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

from tqdm import tqdm

llm_config = LocalInferenceGenerator.LLMConfig(
    model="/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-8B",
    tensor_parallel_size=1,
)
sampling_config = LocalInferenceGenerator.SamplingConfig()
ray_config = LocalInferenceGenerator.RayConfig(num_actors=4, num_gpus=4)
base_engine = LocalInferenceGenerator(
    llm_config=llm_config,
    sampling_config=sampling_config,
    ray_config=ray_config,
)


llm_config = LocalInferenceGenerator.LLMConfig(
    model="/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen2.5-72B",
    tensor_parallel_size=4,
)
sampling_config = LocalInferenceGenerator.SamplingConfig()
ray_config = LocalInferenceGenerator.RayConfig(num_actors=1, num_gpus=4)
synth_engine = LocalInferenceGenerator(
    llm_config=llm_config,
    sampling_config=sampling_config,
    ray_config=ray_config,
)


llm_config = LocalInferenceGenerator.LLMConfig(
    model="/workspace/mnt/lxb_work/hf_dir/hf_model/Qwen/Qwen3-32B",
    tensor_parallel_size=2,
)
sampling_config = LocalInferenceGenerator.SamplingConfig()
ray_config = LocalInferenceGenerator.RayConfig(num_actors=2, num_gpus=4)
label_engine = LocalInferenceGenerator(
    llm_config=llm_config,
    sampling_config=sampling_config,
    ray_config=ray_config,
)


formator = WildGuardFormator()
parser = WildGuardParser()
confidence_filter = ConfidenceFilter()


# for loader in [aegis_v1_loader, aegis_v2_loader, beavortails_loader, toxicchat_loader, wildguard_loader]:
for loader in [wildguard_loader]:
    
    logger.info(f"Loading {loader.name} dataset")
    data = loader.load()
    # data = random.sample(data, 1000) # 如果需要随机采样，建议在这里开启
    logger.info(f"{loader.name} dataset loaded, size={len(data)}")

    logger.info(f"Formatting {loader.name} dataset")
    queries = formator.format(data=data, split=loader.name)
    logger.info(f"{loader.name} formatted, queries={len(queries)}")
    
    # 定义中间文件名
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    cache_file = f"{temp_dir}/confidence_inference_{loader.name}.jsonl"
    generation_result: GenerationList = []

    # --- 检查缓存是否存在 ---
    if os.path.exists(cache_file):
        logger.info(f"Found existing cache: {cache_file}, loading...")
        num_lines = sum(1 for _ in open(cache_file, 'r', encoding='utf-8'))
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=num_lines, desc="Loading Cache"):
                item_dict = json.loads(line)
                
                # 特别注意：还原嵌套的 GenerationCandidate 对象
                candidates = [
                    GenerationCandidate(**cand) for cand in item_dict.get("candidates", [])
                ]
                item_dict["candidates"] = candidates
                
                # 还原成 GenerationResult 对象
                generation_result.append(GenerationResult(**item_dict))
        logger.info(f"Successfully loaded {len(generation_result)} items from cache.")
        
    else:

        logger.info(f"No cache found. Starting pipeline for {loader.name}")

        logger.info(f"Generating {loader.name} dataset")
        inference_result = base_engine.generate(queries=[query.conversations for query in queries])
        logger.info(f"{loader.name} generation done, results={len(inference_result)}")

        for raw, infer in zip(queries, inference_result):
            generation_result.append(
                GenerationResult(
                    id=raw.id,
                    split=raw.split,
                    prompt=raw.prompt,
                    response=raw.response,
                    conversations=raw.conversations,
                    prompt_harm_label=raw.prompt_harm_label,
                    response_harm_label=raw.response_harm_label,
                    response_refusal_label=raw.response_refusal_label,
                    category=raw.category,
                    extra=raw.extra,
                    candidates=[
                        GenerationCandidate(
                            text=cand["text"],
                            token_confidence=cand["token_confidence"],
                        ) for cand in infer
                    ],
                )
            )

        # 保存生成的结果以备下次直接加载
        with open(cache_file, "w", encoding="utf-8") as f:
            for gi in generation_result:
                f.write(json.dumps(asdict(gi), ensure_ascii=False) + "\n")
        logger.info(f"Results saved to {cache_file}")


    logger.info(f"Filtering {loader.name} dataset")
    filtered_result = confidence_filter.apply(generation_result)
    logger.info(f"{loader.name} filtered, kept={len(filtered_result)}")
    # logger.debug([item.confidence for item in filtered_result])


    logger.info("Synth starting...")
    filtered_safe_items = [item for item in filtered_result if item.prompt_harm_label == 0]
    filtered_unsafe_items = [item for item in filtered_result if item.prompt_harm_label == 1]

    safe_by_category = {}
    for item in filtered_safe_items:
        cat = getattr(item, "category", "safe")
        if cat not in safe_by_category:
            safe_by_category[cat] = []
        safe_by_category[cat].append(item)

    logger.info(
        f"{loader.name} safe items: total={len(filtered_safe_items)}, "
        f"by_category={{ {', '.join(f'{k}: {len(v)}' for k, v in safe_by_category.items())} }}"
    )

    unsafe_by_category = {}
    for item in filtered_unsafe_items:
        cat = getattr(item, "category", "unsafe")
        if cat not in unsafe_by_category:
            unsafe_by_category[cat] = []
        unsafe_by_category[cat].append(item)

    logger.info(
        f"{loader.name} unsafe items: total={len(filtered_unsafe_items)}, "
        f"by_category={{ {', '.join(f'{k}: {len(v)}' for k, v in unsafe_by_category.items())} }}"
    )

    # 创建缓存文件夹
    synth_cache_dir = f"{temp_dir}/synth_cache_{loader.name}"
    os.makedirs(synth_cache_dir, exist_ok=True)

    def get_cache_path(category, label_type):
        # 处理类别名中的特殊字符，防止文件名非法
        cat_name = re.sub(r'[^\w\-_\. ]', '_', category)
        return os.path.join(synth_cache_dir, f"synth_{label_type}_{cat_name}.json")

    synthesizer = SelfInstruct(engine=synth_engine)
    synth_safe_instructions = {}
    for i, cat in enumerate(tqdm(safe_by_category, desc="🔐 Processing Safe Categories")):
        cache_path = get_cache_path(cat, "safe")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading cached safe synth for [{cat}]")
            with open(cache_path, "r", encoding="utf-8") as f:
                synth = json.load(f)
        else:
            logger.warn(f"Safe Category {i+1}/{len(safe_by_category)}: {cat}")
            seed_prompts = [item.prompt for item in safe_by_category[cat]]
            synth = synthesizer.self_instruct(seed_prompts, cat)
            # 立即保存该类别的结果
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(synth, f, ensure_ascii=False)
        
        synth_safe_instructions[cat] = synth
        logger.info(f"{loader.name} synth safe [{cat}]: num={len(synth)}")

    synth_unsafe_instructions = {}
    for i, cat in enumerate(tqdm(unsafe_by_category, desc="🚨 Processing Unsafe Categories")):
        cache_path = get_cache_path(cat, "unsafe")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading cached unsafe synth for [{cat}]")
            with open(cache_path, "r", encoding="utf-8") as f:
                synth = json.load(f)
        else:
            logger.warn(f"Unsafe Category {i+1}/{len(unsafe_by_category)}: {cat}")
            seed_prompts = [item.prompt for item in unsafe_by_category[cat]]
            synth = synthesizer.self_instruct(seed_prompts, cat)
            # 立即保存
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(synth, f, ensure_ascii=False)
                
        synth_unsafe_instructions[cat] = synth
        logger.info(f"{loader.name} synth unsafe [{cat}]: num={len(synth)}")
    
    # logger.debug(f"{loader.name} synth safe instructions: {synth_safe_instructions}")
    # logger.debug(f"{loader.name} synth unsafe instructions: {synth_unsafe_instructions}")

    synth_instructions = []
    it = 0
    for cat in synth_safe_instructions.keys():
        for synth_safe_instruction in synth_safe_instructions[cat]:
            synth_instructions.append({
                "id": "s-" + str(it),
                "category": cat, 
                "prompt": synth_safe_instruction,
                "prompt_harm_label": 0,
            })
            it += 1
    for cat in synth_unsafe_instructions.keys():
        for synth_unsafe_instruction in synth_unsafe_instructions[cat]:
            synth_instructions.append({
                "id": "s-" + str(it),
                "category": cat, 
                "prompt": synth_unsafe_instruction,
                "prompt_harm_label": 1,
            })
            it += 1

    logger.info(f"{loader.name} synth instructions: total={len(synth_instructions)}")
    # logger.debug(synth_instructions[0])


    logger.info("Generating synth responses...")
    synth_responses = synth_engine.generate([item["prompt"] for item in synth_instructions])
    logger.info(f"Synth responses generated, results={len(synth_responses)}")
    # logger.debug(synth_responses[0])


    logger.info("Concatenating synth data...")
    synth_data: GuardrailDataset = []
    for item, response in zip(synth_instructions, synth_responses):
        synth_data.append(
            GuardrailItem(
                prompt=item["prompt"],
                response=response[0]["text"],
                prompt_harm_label=item["prompt_harm_label"],
                # 这里先不设置 response_harm_label / response_refusal_label，由后续判断模型填充
                response_harm_label=-1,
                response_refusal_label=-1,
                category=item["category"],
                extra={"id": item["id"]},
            )
        )
    logger.info(f"Synth data concatenated, results={len(synth_data)}")
    # logger.debug(synth_data[0])

    
    synth_judgement_prompts = formator.format(synth_data, split="synth-"+loader.name)
    # logger.debug(synth_judgement_prompts[0])

    logger.info(f"Generating synth labels (response_harm_label and response_refusal_label) for {loader.name} dataset")
    synth_judgement = label_engine.generate(
        [item.conversations[0]["content"] for item in synth_judgement_prompts]
    )
    logger.info(f"Synth labels generated, results={len(synth_judgement)}")
    # logger.debug(synth_judgement[0])
    

    synth_judgement_results = parser.parse([judgement[0]["text"] for judgement in synth_judgement])

    for item, judgement in zip(synth_data, synth_judgement_results):
        item.response_harm_label = judgement["response_harm_label"]
        item.response_refusal_label = judgement["response_refusal_label"]

    # --- 识别失败项并进行第二次尝试 (Retry) ---
    # 使用 enumerate 找出那些解析结果为 -1 的索引
    failed_indices = [
        i for i, item in enumerate(synth_data) 
        if item.response_harm_label == -1 or item.response_refusal_label == -1
    ]

    if failed_indices:
        logger.warn(f"共有 {len(failed_indices)} 条判断未解析出标签，可能是因为合成响应超出模型最大长度。尝试第2次生成...")
        
        # 提取失败项对应的 Prompt
        retry_prompts = [synth_judgement_prompts[i].conversations[0]["content"] for i in failed_indices]
        
        # 第二次生成：可以适当调高一点 temperature 增加多样性，或者保持一致
        retry_outputs = label_engine.generate(retry_prompts)
        retry_judgement_results = parser.parse([res[0]["text"] for res in retry_outputs])
        
        # 将重试后的结果填回原数组
        for i, new_judgement in zip(failed_indices, retry_judgement_results):
            # 只有当第二次解析成功时才覆盖（或者直接覆盖，因为最差也就是 -1）
            synth_data[i].response_harm_label = new_judgement["response_harm_label"]
            synth_data[i].response_refusal_label = new_judgement["response_refusal_label"]

        logger.info(f"重新生成完成，成功恢复标签: "
                    f"{sum(1 for i in failed_indices if synth_data[i].response_harm_label != -1)}")

    # 将 GuardrailItem 列表序列化为 JSONL
    with open(f"synth_data_{loader.name}.jsonl", "w", encoding="utf-8") as f:
        for gi in synth_data:
            f.write(json.dumps(asdict(gi), ensure_ascii=False) + "\n")