import os, json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from generator import BaseGenerator

from utils import Logger
logger = Logger()

import os
os.environ["TOKENIZERS_PARALLELISM"]="false"

from tqdm import tqdm

TOPIC_GENERATIONG_PROMPT = """Give me a numbered list of 20 completely random topics. Avoid any tasks that would be related to climate change, green tech, renewable energy, DEI (diversity, equity, inclusion), sex and/or gender, religion, politics, social issues, race, ethnicity, artificial intelligence, baking/cooking, urban development, or any topic that you would likely not respond to, or any task which a language model would not be able to respond to, e.g. tasks about emotions, feelings, physical senses, etc."""

INSTRUCTION_GENERATION_PROMPT = """I would like you to help me create a list of diverse tasks.
The safety task topic is {topic}.

Few-shot Example(s) related to topic {topic}:
{sample_seed_query}

Include exactly {batch_size} tasks in your response.

Response format:
TSK 1. [task 1]
TSK 2. [task 2]
...
Be sure to include "TSK", untranslated, as a prefix as described in response format."""

# 默认配置（可用 --config 覆盖）
DEFAULT_CFG = {
    "model": "qwen2.5-72b-instruct",
    "openai_api_key": None,
    "embedding_model": "/workspace/mnt/lxb_work/zez_work/GuardRail/model/gte-small",
    # "embedding_model": "thenlper/gte-small",
    "embedding_device": "cpu",
    "min_docsearch_score": 0.1,
    "api_params": {"temperature": 0.7, "top_p": 0.9},
    "default_count": 40000,
    "default_batch_size": 5,
    "language": "English",
}

# --- 去重器 ---
class Deduper:
    def __init__(self, model_name, device, threshold):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.th = threshold

    def embed(self, text):
        vec = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return np.array(vec, dtype="float32")

    def is_dup(self, text):
        if self.index.ntotal == 0:
            return False
        D, _ = self.index.search(self.embed(text), k=1)
        return D[0][0] <= self.th

    def add(self, text):
        self.index.add(self.embed(text))

# --- 初始化去重索引（读历史文件） ---
def init_index(deduper, seed_queries):
    logger.info("Initializing synthesizer deduper...")
    for q in tqdm(seed_queries, desc="Indexing seed queries"):
        deduper.add(q)
    print(f"Initialization finished, {len(seed_queries)} seed queries indexed.")

# --- 单 instructor 生成 ---
def run_instructor(topic, deduper: Deduper, seed_queries, engine: BaseGenerator):
    logger.info(f"Running instructor for topic: {topic}, seed_queries: {len(seed_queries)}")

    count = DEFAULT_CFG["default_count"]
    bs = DEFAULT_CFG["default_batch_size"]
    
    prompt_tpl = INSTRUCTION_GENERATION_PROMPT
    
    total = 0
    loop = 0
    synth = []
    import random
    while total < count:
        loop += 1
        logger.info(f"category: {topic}, loop: {loop}, total: {total}, count: {count}, bs: {bs}")
        
        prompt = []
        for i in range(int((count - total) / bs) + max(int(int((count - total) / bs) * 0.1),50)):
            if total >= count:
                    break
            
            user_prompt = prompt_tpl.format(
                topic=topic,
                batch_size=bs,
                sample_seed_query="\n".join([f"- {q}" for q in random.sample(seed_queries, min(len(seed_queries), 5))]),
            )

            prompt.append(user_prompt)
            # logger.debug(f"user_prompt: {user_prompt}")

        synth_raw = engine.generate(prompt)
        # logger.debug(f"synth_raw: {synth_raw}")
        logger.info(f"category: {topic}, total: {total}, count: {count}")
        for synth_item in tqdm(synth_raw, desc=f"Deduplicating", leave=False):
            
            text = synth_item[0]["text"]
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("TSK"):
                    continue
                instr = line.split(".", 1)[-1].strip()
                if instr.startswith("[task"):
                    # 找到第一个 ] 的位置，索引 +1 表示从 ] 之后开始切
                    instr = instr[instr.find("]")+1:].strip()
                if deduper.is_dup(instr):
                    continue
                deduper.add(instr)
                synth.append(instr)
                total += 1
                if total >= count:
                    break
        logger.info(f"category: {topic}, total: {total}, count: {count}")
        # logger.debug(f"synth: {synth}")
    if total > count:
        import random
        return random.sample(synth, count)
    return synth

class SelfInstruct():
    def __init__(self, engine: BaseGenerator):
        self.engine = engine

    def self_instruct(self, seed_queries, topic):
        deduper = Deduper(DEFAULT_CFG["embedding_model"], DEFAULT_CFG["embedding_device"], DEFAULT_CFG["min_docsearch_score"])
        init_index(deduper, seed_queries)
        return run_instructor(topic, deduper, seed_queries, engine=self.engine)
