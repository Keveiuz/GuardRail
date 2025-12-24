# import json
# from tqdm import tqdm

# INPUT_FILE = "/workspace/mnt/lxb_work/zez_work/GuardRail/temp/synth_data_wildguard.jsonl"


# def analyze_new_jsonl_labels(input_file: str):
#     stats = {
#         "total": 0,

#         # 1
#         "all_three_missing": 0,

#         # 2-4
#         "prompt_label_missing": 0,
#         "response_harm_label_missing": 0,
#         "response_refusal_label_missing": 0,

#         # 5-6
#         "exactly_1_missing": 0,
#         "exactly_2_missing": 0,

#         # 7
#         "response_harm_unsafe": 0,

#         # 8-9
#         "response_harm_unsafe_and_refusal_1": 0,
#         "response_harm_unsafe_and_refusal_0": 0,
#     }

#     with open(input_file, "r", encoding="utf-8") as f:
#         for line in tqdm(f, desc="Analyzing labels"):
#             item = json.loads(line)
#             stats["total"] += 1

#             p = item.get("prompt_harm_label", -1)
#             r_h = item.get("response_harm_label", -1)
#             r_r = item.get("response_refusal_label", -1)

#             labels = [p, r_h, r_r]
#             missing_cnt = sum(1 for x in labels if x == -1)

#             # 1. 三个标签都没有
#             if missing_cnt == 3:
#                 stats["all_three_missing"] += 1

#             # 2-4. 单独缺失
#             if p == -1:
#                 stats["prompt_label_missing"] += 1
#             if r_h == -1:
#                 stats["response_harm_label_missing"] += 1
#             if r_r == -1:
#                 stats["response_refusal_label_missing"] += 1

#             # 5-6. 精确缺失数量
#             if missing_cnt == 1:
#                 stats["exactly_1_missing"] += 1
#             elif missing_cnt == 2:
#                 stats["exactly_2_missing"] += 1

#             # 7. response 不安全
#             if r_h == 1:
#                 stats["response_harm_unsafe"] += 1

#                 # 8. 不安全 + 拒绝
#                 if r_r == 1:
#                     stats["response_harm_unsafe_and_refusal_1"] += 1

#                 # 9. 不安全 + 未拒绝
#                 elif r_r == 0:
#                     stats["response_harm_unsafe_and_refusal_0"] += 1

#     return stats


# def print_stats(stats: dict):
#     print("\n========== Label Statistics ==========")
#     for k, v in stats.items():
#         print(f"{k:45s}: {v}")
#     print("======================================\n")


# def main():
#     print(f"Input file: {INPUT_FILE}")
#     stats = analyze_new_jsonl_labels(INPUT_FILE)
#     print_stats(stats)


# if __name__ == "__main__":
#     main()


import json
from collections import Counter
from tqdm import tqdm

INPUT_FILE = "/workspace/mnt/lxb_work/zez_work/GuardRail/temp/synth_data_wildguard.jsonl"


def analyze_label_distributions(input_file: str):
    # 单标签分布
    prompt_label_dist = Counter()
    response_harm_label_dist = Counter()
    response_refusal_label_dist = Counter()

    # 三标签组合分布
    triple_label_dist = Counter()

    total = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Analyzing label distributions"):
            item = json.loads(line)
            total += 1

            p = item.get("prompt_harm_label", -1)
            r_h = item.get("response_harm_label", -1)
            r_r = item.get("response_refusal_label", -1)

            # 单标签统计
            prompt_label_dist[p] += 1
            response_harm_label_dist[r_h] += 1
            response_refusal_label_dist[r_r] += 1

            # 组合统计
            triple_label_dist[(p, r_h, r_r)] += 1

    return {
        "total": total,
        "prompt_label_dist": prompt_label_dist,
        "response_harm_label_dist": response_harm_label_dist,
        "response_refusal_label_dist": response_refusal_label_dist,
        "triple_label_dist": triple_label_dist,
    }


def print_single_label_dist(name: str, dist: Counter):
    print(f"\n--- {name} ---")
    for k in sorted(dist.keys()):
        print(f"  {k:>2}: {dist[k]}")


def print_triple_label_dist(dist: Counter, top_k: int = None):
    print("\n--- Triple Label Combination Distribution ---")
    items = dist.most_common(top_k)
    for (p, r_h, r_r), cnt in items:
        print(f"  (prompt={p:>2}, response_harm={r_h:>2}, response_refusal={r_r:>2}) : {cnt}")


def main():
    print(f"Input file: {INPUT_FILE}")

    stats = analyze_label_distributions(INPUT_FILE)

    print(f"\nTotal samples: {stats['total']}")

    print_single_label_dist("Prompt Harm Label Distribution", stats["prompt_label_dist"])
    print_single_label_dist("Response Harm Label Distribution", stats["response_harm_label_dist"])
    print_single_label_dist("Response Refusal Label Distribution", stats["response_refusal_label_dist"])

    # 如果组合太多，可以只看 top-k
    print_triple_label_dist(stats["triple_label_dist"], top_k=50)


if __name__ == "__main__":
    main()
