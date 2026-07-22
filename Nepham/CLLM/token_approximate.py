import json
import math
from pathlib import Path
from transformers import AutoTokenizer

MODEL = Path(
    "/share/project/wuhaiming/spaces/CLLM/models/Qwen3-4B-Base-Char"
)
DATA = Path(
    "/share/project/wuhaiming/spaces/CLLM/data/processed/cpt/"
    "tiger_pretrain_zh_train.jsonl"
)

WORLD_SIZE = 8
PER_DEVICE_BATCH = 8
GRAD_ACC = 8
MAX_LENGTH = 1024
TARGET_TOKENS = 2_000_000_000

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True,
    use_fast=True,
)

lengths = []

with DATA.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue

        item = json.loads(line)

        if "messages" in item:
            texts = [
                msg["content"]
                for msg in item["messages"]
                if msg.get("role") == "assistant"
            ]
            text = "\n".join(texts)
        elif "text" in item:
            text = item["text"]
        else:
            raise ValueError(
                f"line {line_no}: unsupported keys {list(item)}"
            )

        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        lengths.append(len(token_ids))

lengths.sort()
n = len(lengths)
mean_len = sum(lengths) / n
global_batch = WORLD_SIZE * PER_DEVICE_BATCH * GRAD_ACC
tokens_per_step = global_batch * mean_len
target_steps = math.ceil(TARGET_TOKENS / tokens_per_step)

def percentile(p):
    return lengths[min(int((n - 1) * p), n - 1)]

print(f"samples              : {n:,}")
print(f"mean tokens/sample   : {mean_len:.2f}")
print(f"p50 tokens/sample    : {percentile(0.50)}")
print(f"p90 tokens/sample    : {percentile(0.90)}")
print(f"p95 tokens/sample    : {percentile(0.95)}")
print(f"p99 tokens/sample    : {percentile(0.99)}")
print(f"truncated ratio      : {sum(x >= MAX_LENGTH for x in lengths) / n:.4%}")
print(f"global batch         : {global_batch}")
print(f"approx tokens/step   : {tokens_per_step:,.0f}")
print(f"tokens at step 4000  : {tokens_per_step * 4000:,.0f}")
print(f"tokens at step 4420  : {tokens_per_step * 4420:,.0f}")
print(f"tokens at step 30000 : {tokens_per_step * 30000:,.0f}")
print(f"steps for 2B tokens  : {target_steps:,}")
