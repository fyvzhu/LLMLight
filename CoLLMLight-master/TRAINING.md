# 训练、最佳模型导出与 CityFlow 测试

从 `CoLLMLight-master/` 执行下列命令，固定使用现有 uv 环境 `.venv/bin/python`。本项目使用已有 TRL 0.9.2 完成 LoRA-SFT 和 LoRA-PPO，不需要 LLaMA-Factory。第二轮迁移没有重建环境，也没有安装或升级依赖；历史版本调整保留在工作区父目录 `new_requirements.txt`。

## 目录

- `llm_models/`：原始基础模型权重；`models/`：原有智能体 Python 代码，两者不可混淆。
- `CityFlow/`：仿真器源码，当前环境已安装其扩展。
- `my_tests/`：新增测试、真实执行入口、日志和报告；`tests/`：原有回归测试。
- `outputs/<run_name>/`：每次实验的数据、Adapter、checkpoint、验证结果和完整最佳模型。

## 唯一配置来源

`.env` / `.env.example` 仅保留教师接口、密钥、教师生成参数，以及学生基础模型/通用加载设置。`.env` 的密钥不进入 YAML、快照或日志。

`config/config.yaml` 管理实验、数据抽样、SFT、PPO、LoRA、导出和 CityFlow 测试。旧 `ppo_config.yaml` 已移除，有效参数不再重复；历史参考存为 `my_tests/legacy_ppo_config.txt`。

- 模型设置：显式覆盖 > 进程环境变量 > `.env` > 默认值。
- YAML 管理设置：显式覆盖 > 指定 YAML > 默认值；旧的 `PPO_BATCH_SIZE` 等环境变量不再覆盖 YAML。
- 所有相对路径均相对于项目根目录，`--config` 和 `--env-file` 也遵循此规则。
- `.env` 中 `STUDENT_MODEL_PATH=./llm_models/Qwen2.5-1.5B-Instruct` 是训练起点。
- YAML 的 `cityflow_test.model_path` 是测试使用的完整模型，和训练起点不同；`null` 自动定位当前实验的 `trained_best_model/`。
- YAML 拼错字段、重复键和无效类型会报错，不会默默忽略。

## 逐阶段执行

```bash
# 原始数据已存在时不重复采样，采样入口拒绝覆盖已有文件
.venv/bin/python my_tests/run_real_stages.py sample
# 教师最多处理 teacher_data.max_samples 条记录，每条正常发起 ATR/RA 两次请求
.venv/bin/python my_tests/run_real_stages.py teacher
.venv/bin/python my_tests/run_real_stages.py prepare
.venv/bin/python my_tests/run_real_stages.py sft
.venv/bin/python my_tests/run_real_stages.py merge
.venv/bin/python my_tests/run_real_stages.py ppo
# 需要时从 checkpoint 恢复；演示入口从 update_1 再更新一次
.venv/bin/python my_tests/run_real_stages.py resume
# 或指定任意已保存的 checkpoint
.venv/bin/python ppo_ft.py --resume outputs/<run_name>/ppo_checkpoints/update_1 --additional-updates 1
```

教师断点恢复不重复调用已完成样本；改变教师输入或参数时使用新实验名。历史教师数据可以复用，但必须同时使用对应的 `selected_samples.json`。第二轮真实训练复测配置是 `my_tests/configs/training.yaml`，通过 `--config my_tests/configs/training.yaml` 传入；这份配置只用于独立复测，主配置保持默认实验。

训练结束时默认 `export.auto_export: true`，自动导出最佳完整模型；设为 false 可只训练，稍后独立导出。

## 权重的关系

```text
原始 llm_models/<model>/ + sft_adapter/
    → 合并得到 sft_merged/（完整 SFT 模型）
    + ppo_best/（最佳 PPO Adapter）
    → 合并得到 trained_best_model/（完整最终模型）
```

`ppo_checkpoints/update_N/` 用于恢复训练；`ppo_best/` 保存验证集选出的最佳 PPO Adapter，不一定是最后一次更新。SFT 当前保存结束时权重；这里 best 指本次 SFT 起点上的最佳验证 PPO checkpoint，不宣称已实现 SFT 多 checkpoint 最优选模。

## 最佳完整模型导出

```bash
.venv/bin/python export_best_model.py --config config/config.yaml
```

默认输出：`outputs/<run_name>/trained_best_model/`。包含完整 safetensors 权重、模型配置、tokenizer、generation config、来源清单与导出验证报告。可直接 `AutoModelForCausalLM.from_pretrained(path, local_files_only=True)`，不需要 PEFT、不再组合外部 Adapter。

导出会核对 best checkpoint 的权重与分数，比较 FP32 合并误差及保存精度误差，独立加载导出模型，并在同一验证记录上重新生成 ATR/RA。验证决策发生变化会报错并保存失败报告，不会把未验证的半成品发布为最终模型。相同来源的已完成导出可校验后复用；来源不同则拒绝静默覆盖，应指定新的 `export.output_dir`。

首次导出使用非量化权重安全合并，`export.device` 与 `export.dtype` 可配置。auto 精度沿用 SFT 模型保存精度；8B 场景需要足够的主存和磁盘。推理不加载 PPO value head；value head/优化器仍在训练 checkpoint 内。

## CityFlow 真模型测试

```yaml
cityflow_test:
  model_path: null  # 当前 outputs/<run_name>/trained_best_model
  backend: transformers
  dataset: synthetic_4x4
  traffic_file: anon_4_4_synthetic_8000.json
  run_counts: 60
  max_new_tokens: 1024
  seed: 42
```

也可将 `model_path` 改成其他完整模型目录，例如 `./outputs/other_run/trained_best_model`。无效路径或 Adapter-only 目录会明确失败，不回退到原始 Qwen。

```bash
# 实际复用原 CoLLMLight 控制循环与 CityFlow
.venv/bin/python run_CoLLMlight.py --config config/config.yaml
# 显式路径优先于 YAML
.venv/bin/python run_CoLLMlight.py --model-path ./outputs/qwen15b_smoke/trained_best_model
# 检查至少发生一次合法模型决策且没有兜底动作
.venv/bin/python my_tests/run_cityflow_model_smoke.py --config config/config.yaml
```

新统一入口使用已验证的本地 Transformers 后端，无需先启动服务。原 HTTP 客户端代码仍保留，但远程 vLLM/lmdeploy 服务不属于本轮验证范围，不能把仅配置模型路径等同于服务已部署。济南/杭州/纽约可通过 YAML 选择对应 dataset 与实际存在的 traffic_file；设置正式时长后另行运行基准，默认 60 秒只证明集成可运行。

结果在 `outputs/<run_name>/cityflow_test/`，包含实际模型路径、LLM 调用次数、动作解析及交通指标。

## 回归与日志

```bash
.venv/bin/python -m unittest discover -s my_tests -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python my_tests/run_real_stages.py reload
```

普通 unittest 不发起付费调用或启动训练。真实阶段的日志位于 `my_tests/logs/<run_name>/real_<stage>.log`；第二轮汇总见 `my_tests/第二轮测试结果.md`。旧报告作为历史记录保留，旧路径当前位置见 `my_tests/migration_report.json`。

默认仍是小样本冒烟训练。正式训练请使用新的 run_name，扩大数据并解除 max_steps/max_updates 限制。数据按记录划分训练/验证，ATR/RA 不跨集合；当前未新增独立测试集划分、全量 benchmark、多卡或 4bit 验证。小样本跑通不代表交通控制性能达标。
