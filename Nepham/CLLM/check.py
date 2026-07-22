import os
import json
import torch

ckpt = "/share/project/wuhaiming/spaces/CLLM/adapters/char-cpt/v4-20260721-004711/checkpoint-2000"

optimizer_path = os.path.join(ckpt, "optimizer.pt")
state = torch.load(optimizer_path, map_location="cpu", weights_only=False)

print("checkpoint param groups:", len(state["param_groups"]))
print("params per group:", [len(g["params"]) for g in state["param_groups"]])

with open(os.path.join(ckpt, "args.json")) as f:
    args = json.load(f)

for key in [
    "tuner_type", "tuner_backend", "target_modules",
    "lora_rank", "lora_alpha", "lorap_lr_ratio",
    "optim", "optimizer", "use_galore", "freeze_parameters"
]:
    print(key, "=", args.get(key))