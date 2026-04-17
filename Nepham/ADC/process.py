import difflib
import json
import os
import random
from typing import List, Dict, Any


INS = """请检测待纠错句子中的**中文拼写错误**。

**纠错指南**
- 聚焦于**拼写错误（字音或字形）**，无需关注语法，更不要擅自优化语句表达
- 原句中可能有零处、一处或多处错误，**不要过度纠正**
- 如果没有错误，则输出原句
- 如果有错误，则将每一处错误汉字替换为正确的，然后将修改后的正确句子输出
- 除此之外不要输出任何其他内容

---
下面是可供参考的示例，每个示例中**第一行是原句，第二行是正确句**：

示例 1：
今天天汽真不搓。
今天天气真不错。

示例 2：
我要吃早惨。
我要吃早餐。

示例 3：
今年是我的本命年。
今年是我的本命年。

---

现在请对以下句子进行纠错：
"""


def load_raw_samples(raw_data_path: str):
    samples = []
    with open(raw_data_path, "r", encoding="utf-8") as f:
        if raw_data_path.endswith(".jsonl"):
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                src = item.get("input", "").strip()
                tgt = item.get("output", "").strip()
                if src and tgt:
                    samples.append((src, tgt))
        elif raw_data_path.endswith(".json"):
            data = json.load(f)
            for item in data:
                src = item.get("input", "").strip()
                tgt = item.get("output", "").strip()
                if src and tgt:
                    samples.append((src, tgt))
        elif raw_data_path.endswith(".txt"):
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                src = parts[0].strip()
                tgt = parts[1].strip()
                if src and tgt:
                    samples.append((src, tgt))
        else:
            raise ValueError(f"不支持的文件类型: {raw_data_path}")
    return samples


def build_item(src: str, tgt: str) -> Dict[str, Any]:
    return {
        "src": src,
        "tgt": tgt
    }


def process_dataset_with_error_ratio(
    raw_data_path: str,
    strategy: str = None,  # 可选: "downsample" / "oversample"
    error_ratio: float = 0.5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    random.seed(seed)

    raw_samples = load_raw_samples(raw_data_path)

    if strategy is None:
        processed_data = []
        for src, tgt in raw_samples:
            item = build_item(src, tgt)
            processed_data.append(item)
        return processed_data

    error_samples = []
    clean_samples = []
    for src, tgt in raw_samples:
        item = build_item(src, tgt)
        if src != tgt:
            error_samples.append(item)
        else:
            clean_samples.append(item)

    print(f"原始错误样本数: {len(error_samples)}")
    print(f"原始正确样本数: {len(clean_samples)}")

    if error_ratio <= 0 or error_ratio >= 1:
        raise ValueError("error_ratio 必须在 (0, 1) 之间")

    # 目标关系：
    # error / total = error_ratio
    # => error / (error + clean) = error_ratio
    # => clean = error * (1-error_ratio) / error_ratio
    target_clean_for_error = int(len(error_samples) * (1 - error_ratio) / error_ratio)
    target_error_for_clean = int(len(clean_samples) * error_ratio / (1 - error_ratio))

    if strategy == "downsample":
        # 尽量不重复采样，按较小一侧裁剪
        target_error = min(len(error_samples), target_error_for_clean)
        target_clean = int(target_error * (1 - error_ratio) / error_ratio)

        if target_clean > len(clean_samples):
            target_clean = len(clean_samples)
            target_error = int(target_clean * error_ratio / (1 - error_ratio))

        sampled_error = random.sample(error_samples, target_error)
        sampled_clean = random.sample(clean_samples, target_clean)

    elif strategy == "oversample":
        # 少的一类通过重复采样补足
        if target_clean_for_error <= len(clean_samples):
            sampled_error = error_samples
            sampled_clean = random.sample(clean_samples, target_clean_for_error)
        else:
            sampled_error = random.sample(error_samples, target_error_for_clean)
            sampled_clean = clean_samples

        # 如果某一类不足则重复采样
        desired_total = len(sampled_error) + len(sampled_clean)
        desired_error = int(desired_total * error_ratio)
        desired_clean = desired_total - desired_error

        if len(sampled_error) < desired_error:
            extra = random.choices(error_samples, k=desired_error - len(sampled_error))
            sampled_error = sampled_error + extra
        if len(sampled_clean) < desired_clean:
            extra = random.choices(clean_samples, k=desired_clean - len(sampled_clean))
            sampled_clean = sampled_clean + extra
    else:
        raise ValueError("strategy 只能是 'downsample' 或 'oversample'")

    processed_data = sampled_error + sampled_clean
    random.shuffle(processed_data)

    final_error = sum(1 for x in processed_data if x["src"] != x["tgt"])
    final_total = len(processed_data)
    print(f"最终总样本数: {final_total}")
    print(f"最终错误样本数: {final_error}")
    print(f"最终错误样本比例: {final_error / final_total:.4f}")

    return processed_data


if __name__ == "__main__":
    data1 = process_dataset_with_error_ratio(
        "/share/project/wuhaiming/spaces/LlamaFactory/data/twnlp_csc.jsonl",
        # error_ratio=0.99,
        # seed=42,
        # strategy="downsample"
    )

    data2 = process_dataset_with_error_ratio(
        "/share/project/wuhaiming/spaces/ADC/data/sighan/SIGHAN-train.txt",
        # error_ratio=0.95,
        # seed=42,
        # strategy="downsample"
    )

    all_data = data1+data2
    # random.shuffle(all_data)

    final_error = sum(1 for x in all_data if x["src"] != x["tgt"])
    print(f"混合后总数: {len(all_data)}")
    print(f"混合后错误比例: {final_error / len(all_data):.4f}")

    with open("../dataset/csc_mix.jsonl", "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")