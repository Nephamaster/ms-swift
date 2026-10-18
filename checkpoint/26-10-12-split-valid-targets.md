# split 预训练零有效目标分片过滤

## 背景与边界

用户提供的 ALIGN/CPT 日志分析指出验证 loss 持续为 NaN，最终样本中存在长度为 1 的分片。
当前模板先编码再切分，每个分片的首个 label 被设为 -100；例如 2049 tokens 会切成
2048 + 1，最后一个分片在 causal shift 后没有有效预测目标。

本次修复消除这一输入条件，但没有使用远端验证数据及模型权重复现历史 NaN，不能据此
断言这是全部 NaN 的唯一原因，也不能断言已有 checkpoint 的参数损坏。

## 实现

- 在 `swift/dataset/utils.py` 的 `EncodePreprocessor.preprocess()` 中，完成模板编码及
  切分后，只在 `causal_lm + split` 路径保留 `labels[1:]` 至少包含一个非 -100 标签的分片。
- 同时处理未触发切分的原始短样本、长文档的无效尾段及全部标签被屏蔽的分片；保留有效
  分片的内容、元数据和顺序。返回空列表时使用现有 RowPreprocessor 的样本删除机制。
- 训练集和验证集都经过此入口；非 split 策略和其他任务维持原行为。
- EncodePreprocessor 使用独立的 `cache_format_version` 后缀 `-encode-v1`，使显式
  map-cache 文件名与旧版本不同。此修复不重新清洗通过 cached_dataset/cached_val_dataset
  直接加载的既有导出数据。
- 未修改两个训练脚本及用户原有改动；未改变 loss 聚合方式。

## 本地验证

`tests/general/test_encode_preprocessor_targets.py` 包含 7 项测试，覆盖实际模板切分方法的
2048/2049/2050/4097 tokens 边界、无效原始短样本、全部无效分片、有效分片元数据与顺序、
非目标路径、AddLengthPreprocessor 返回契约及旧缓存文件名隔离。

本地缺少 torch、datasets、modelscope、transformers，采用 AST 提取当前源码中的真实
`Template._encode_truncated`、`_get_length`、预处理器类与缓存路径方法，隔离依赖后执行
上述测试：7 项通过。未验证完整 Dataset.map、多进程缓存流程、真实模型前向或远端验证 loss。
Python 执行后已清理项目内的 __pycache__。

## 远端应用与验收

1. 确保训练虚拟环境实际加载的是更新后的源码；此前日志来自 site-packages/swift，
   仅同步仓库而不更新实际安装可能不生效。
2. 首次验证建议在两个脚本中临时使用 `--load_from_cache_file false`，强制重新处理。
   若使用 cached_dataset/cached_val_dataset，须重新导出或在最终标签上显式过滤，不能仅改开关。
3. 在依赖齐全的环境运行 `python -B -m unittest discover -s tests/general
   -p test_encode_preprocessor_targets.py -v`（写成一行执行），运行后清理项目内 __pycache__。
4. 检查最终训练/验证样本均满足 `any(label != -100 for label in labels[1:])`，再使用
   保留的 checkpoint 在完整验证集上确认 loss 有限。若仍有 NaN，捕获异常 batch 的标签和
   logits，继续排查参数及数值问题。

## 后续：启动验证仍为 NaN

用户随后提供 `epoch=0`、`global_step=0/6289` 的启动验证结果：`eval_loss=NaN`，
`eval_token_acc=0.44204692`。当前本地 ALIGN 脚本已设置 `load_from_cache_file=false` 和
`eval_on_start=true`。远端实际导入源码位置尚未确认。

因此不能把单 token 尾段视为已确认的历史 NaN 根因。step 0 只说明异常出现在本轮参数更新前，
不能排除初始模型参数、前向计算、标签或 loss 路径的问题。

新增 `Nepham/CCLLM/diagnose_eval_nan.py`，通过以下参数临时启用：

```bash
--external_plugins Nepham/CCLLM/diagnose_eval_nan.py \
--eval_on_start true \
```

诊断沿用当前源码及 ms-swift v4.4.2 的 `Seq2SeqTrainer.compute_loss` 接口，不替换 loss。
首次遇到非有限验证 loss，会在实际 output_dir 写入 `nan_diagnostic_rankN.json` 和 `.pt`，
然后抛出异常停止；即使完整验证返回，也会主动抛出异常停止，防止进入完整训练。
移除插件参数后恢复正常运行。`.pt` 只包含该 batch 的张量输入，不包含模型参数或 logits。

JSON 包含实际包版本、预处理器文件位置和源码、模型类与源码位置、每行 causal shift 后的
有效标签数、logits/loss_scale 中的非有限值数量。logits 与 labels 长度一致且未启用
logits_to_keep 时，另外分块用 FP32 对有效目标计算参考交叉熵；无法确认对齐时明确跳过。

- 有效目标为 0：检查实际加载补丁、最终 collator 标签以及缓存/数据路径。
- 有效目标大于 0，logits 含非有限值：进一步定位模型参数与前向计算。
- 有效目标大于 0，logits 有限，参考 CE 有限而原 loss 非有限：检查实际 loss 函数、权重和归一化。
- 全部单 batch loss 有限而汇总仍非有限：继续检查评估聚合过程。

本地完成语法检查及 4 项控制流隔离检查：正常返回、异常 batch 标签快照、训练路径透传、
验证后停止与重复安装保护。新增 `tests/general/test_eval_nan_diagnostic.py` 的 5 项 PyTorch
数值测试，因本地缺少 PyTorch 未执行。可在远端运行：

```bash
python -B -m unittest discover -s tests/general -p test_eval_nan_diagnostic.py -v
```

没有远端模型前向结果前，本次新增属于诊断能力，不是已验证的 NaN 修复。

## 14:15 启动日志：确认异常 batch 与实际加载源码

依据 `semantic_alignment_20261012_141549.log`：

- 插件路径已修正，成功加载；启动命令使用 `load_from_cache_file=false`，Map 完成。
- 训练数据由 543894 条原始记录切成 804890 条，验证数据由 566 条切成 852 条；两者
  `min_length=1`。日志中的 `Dataset filtered` 是条数变化的通用提示，不代表有效目标过滤已经执行。
- 启动验证约进行到 61/107 时，rank 3 在 global_step=0 捕获 NaN。labels 为 `[1, 1]`，
  有效 shifted targets 为 0，logits 为 `[1, 1, 154019]` 且无非有限值，模型返回的 loss 为 NaN。
- 该进程报告 `swift_version=4.4.2`，实际加载 `env/lib/python3.12/site-packages/swift/dataset/utils.py`；
  `preprocessor_source` 仍只有 `return self.template.encode(row, return_length=True)`。
  因此该次运行确实没有执行此前源码中的有效目标过滤。这是进程快照，不能用于断言远端仓库当前版本。

这次捕获的 batch 已确认零监督目标，且旧预处理函数绕过了此前修复；仍未证明所有历史 NaN
均只有这一原因，也未完成补丁生效后的完整验证。

`Nepham/CCLLM/apply_nan_filter_patch.py` 可由远端训练环境的 Python 执行，将仓库中已修改的
单个方法同步到实际安装文件，保留其他实现和包版本，并写入 `.before-nan-filter` 备份。
本地使用旧方法的最小源码样本验证了 6 个行为案例、重复应用幂等性和相邻类保持不变，全部通过；
没有修改本地或远端已安装包。该工具不迁移缓存版本，应用后首次运行必须使用
`load_from_cache_file=false`，预导出的 cached_dataset 仍需重新生成。
