import random
import json
from dataclasses import asdict

from dataloader import AegisV1Loader, AegisV2Loader, BeavorTailsLoader, ToxicChatLoader, WildGuardLoader
from generator import AliyunGenerator
from formator import WildGuardFormator
from parser import WildGuardParser
from filter import ConfidenceFilter
from synthesizer import SelfInstruct

from utils import GenerationCandidate, GenerationResult, GenerationList, GuardrailItem, GuardrailDataset
from utils import Logger
logger = Logger()

aegis_v1_loader = AegisV1Loader()
aegis_v2_loader = AegisV2Loader()
beavortails_loader = BeavorTailsLoader()
toxicchat_loader = ToxicChatLoader()
wildguard_loader = WildGuardLoader()

engine = AliyunGenerator()
formator = WildGuardFormator()
parser = WildGuardParser()
confidence_filter = ConfidenceFilter()


# for loader in [aegis_v1_loader, aegis_v2_loader, beavortails_loader, toxicchat_loader, wildguard_loader]:
for loader in [wildguard_loader]:

    logger.info(f"Loading {loader.name} dataset")
    data = loader.load()
    data = random.sample(data, 10)
    logger.info(f"{loader.name} dataset loaded, size={len(data)}")

    logger.info(f"Formatting {loader.name} dataset")
    queries = formator.format(data=data, split=loader.name)
    logger.info(f"{loader.name} formatted, queries={len(queries)}")

    logger.info(f"Generating {loader.name} dataset")
    inference_result = engine.generate(
        [query.conversations[0]["content"] for query in queries],
        model="qwen3-8b",
        logprobs=True,
        temperature=0.7,
        top_p=0.9)
    logger.info(f"{loader.name} generation done, results={len(inference_result)}")

    generation_result: GenerationList = []
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
                        text=infer["response"],
                        token_confidence=infer["token_confidence"],
                    )
                ],
            )
        )

    logger.info(f"Filtering {loader.name} dataset")
    filtered_result = confidence_filter.apply(generation_result)
    logger.info(f"{loader.name} filtered, kept={len(filtered_result)}")
    logger.debug([item.confidence for item in filtered_result])
    



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

    synth_safe_instructions = {}
    for cat in safe_by_category:
        self_instruct = SelfInstruct()
        # 只把文本 prompt 作为 seed 传入自指令生成器
        seed_prompts = [item.prompt for item in safe_by_category[cat]]
        synth = self_instruct.self_instruct(seed_prompts, cat)
        synth_safe_instructions[cat] = synth
        logger.info(f"{loader.name} synth safe instructions for [{cat}]: num={len(synth)}")

    synth_unsafe_instructions = {}
    for cat in unsafe_by_category:
        self_instruct = SelfInstruct()
        seed_prompts = [item.prompt for item in unsafe_by_category[cat]]
        synth = self_instruct.self_instruct(seed_prompts, cat)
        synth_unsafe_instructions[cat] = synth
        logger.info(f"{loader.name} synth unsafe instructions for [{cat}]: num={len(synth)}")
    
    logger.debug(f"{loader.name} synth safe instructions: {synth_safe_instructions}")
    logger.debug(f"{loader.name} synth unsafe instructions: {synth_unsafe_instructions}")

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
    logger.debug(synth_instructions)



    logger.info("Generating synth responses...")
    synth_responses = engine.generate([item["prompt"] for item in synth_instructions], model="qwen2.5-72b-instruct")
    logger.info(f"Synth responses generated, results={len(synth_responses)}")
    logger.debug(synth_responses[0])


    logger.info("Concatenating synth data...")
    synth_data: GuardrailDataset = []
    for item, response in zip(synth_instructions, synth_responses):
        synth_data.append(
            GuardrailItem(
                prompt=item["prompt"],
                response=response or "",
                prompt_harm_label=item["prompt_harm_label"],
                # 这里先不设置 response_harm_label / response_refusal_label，由后续判断模型填充
                response_harm_label=-1,
                response_refusal_label=-1,
                category=item["category"],
                extra={"id": item["id"]},
            )
        )
    logger.info(f"Synth data concatenated, results={len(synth_data)}")
    logger.debug(synth_data[0])


    synth_judgement_prompts = formator.format(synth_data, split="synth-"+loader.name)

    logger.info(f"Generating synth labels (response_harm_label and response_refusal_label) for {loader.name} dataset")
    synth_judgement = engine.generate(
        [item.conversations[0]["content"] for item in synth_judgement_prompts],
        model="qwen2.5-72b-instruct",
        temperature=0.7,
        top_p=0.9)
    logger.info(f"Synth labels generated, results={len(synth_judgement)}")
    logger.debug(synth_judgement[0])

    synth_judgement_results = parser.parse(synth_judgement)

    for item, judgement in zip(synth_data, synth_judgement_results):
        item.response_harm_label = judgement["response_harm_label"]
        item.response_refusal_label = judgement["response_refusal_label"]

    # 将 GuardrailItem 列表序列化为 JSONL
    with open(f"synth_data_{loader.name}.jsonl", "w", encoding="utf-8") as f:
        for gi in synth_data:
            f.write(json.dumps(asdict(gi), ensure_ascii=False) + "\n")
