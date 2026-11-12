# CoLLMLight 训练与评测

本项目采用 ICLR 2026 发布代码的 ATR/RA 教师生成、PPO 及 CityFlow 控制流程。补充的部分是 LoRA-SFT、阶段衔接、统一配置、日志和完整最佳模型导出；原代码中的批量计数、Adapter 恢复和验证选模错误保留必要修复。没有旧版教师格式兼容层，没有额外实验训练入口，也不输出基线比对 JSON。

## 环境与配置

使用现有 `CoLLMLight-master/.venv`。教师和学生模型连接设置在 `.env`；训练与评测参数统一在 `config/config.yaml`。只修改这一份主配置即可调整数据量和训练轮次，不需要建立“调通专用”配置。

输出根目录为顶层 `output_root`，训练随机种子在 `training.random_seed`。算法名称固定为CoLLMLight，不在YAML中配置，也没有experiment分组或实验模式开关。教师数据抽样和CityFlow评测分别使用其各自的random_seed/seed。

```yaml
output_root: ./outputs
training:
  random_seed: 42
  enable_ppo: true
  data_preparation: sample_and_generate
```

当前配置数值有限，以适配现有资源：采样3个决策步、教师2条、SFT最多2次更新、PPO最多4条记录和2个外层epoch。采样、数据上限、训练更新上限均由YAML控制，不在代码中固定为小规模。正式运行可修改对应参数：sampling.max_steps=0、teacher_data.max_samples=0、sft.max_steps=-1、ppo.max_samples=0，并设置需要的epoch/batch等值。

## 两个入口

```bash
# 完整训练；默认读取config/config.yaml
.venv/bin/python training/collmlight/train_collmlight.py

# 仅评测，不自动训练；加载本次训练导出的完整最佳模型
.venv/bin/python run_CoLLMlight.py --run-dir outputs/CoLLMLight/CoLLMLight_年月日_时分秒_微秒
```

也可在主配置的 `cityflow_test.model_path` 填写完整模型路径，然后直接运行 `run_CoLLMlight.py`。不要同时指定互相冲突的模型路径与run-dir。

`training.enable_ppo: false` 只执行SFT并导出其验证最佳模型；true执行SFT→合并SFT最佳→PPO→导出PPO验证最佳。SFT用最小验证loss，PPO用最大验证RA reward，平分保留较早者；CityFlow测试不参与选模。

## 数据准备模式和输入

| data_preparation | inputs.traffic_samples | inputs.teacher_data |
| --- | --- | --- |
| sample_and_generate | 忽略，自动采样 | 忽略，自动生成 |
| generate_teacher | 必填 | 忽略，自动生成 |
| reuse，启用PPO | 必填，作为PPO场景输入 | 必填 |
| reuse，仅SFT | 忽略 | 必填 |

未启用的输入可设null，不需要额外enable开关。输入数据不会被覆盖。可复用输入分目录存放：

```text
data/FinetuneData/
├── traffic_samples/SynTrain_sample_all.json
└── teacher_data/syn_rt_data.json
```

交通采样含观测、历史、信号效果和Best_Signal，用于教师生成及PPO。教师数据严格采用发布版六字段：atr_user_prompt、atr_output、ra_user_prompt、ra_output、data、complex_level。没有sample_id/teacher_atr_prompt等扩展字段。SFT转换会按交通记录生成内部划分索引，保存在SFT转换结果中，不写回教师原始数据。不接受早期扩展格式，应重新生成。

教师生成采用发布版过滤和随机打乱后的前N条，以及原版easy/complex判断、adaptive ATR后接RA；未自行增加去重排序、多个候选选择等方法。PPO仅跳过缺少历史的初始记录，避免原提示词直接访问history[-1]越界。

## 输出

```text
outputs/CoLLMLight/CoLLMLight_年月日_时分秒_微秒/
├── data/
│   ├── traffic_samples/          # 本次采样或复制的输入
│   ├── teacher_data/             # 原生教师数据
│   ├── sft_data/                 # 必要的SFT转换及训练/验证划分
│   └── ppo_rollouts/             # 原版生成结果及实际token IDs
├── models/
│   ├── sft_checkpoints/
│   ├── sft_best/
│   ├── sft_merged/
│   ├── ppo_checkpoints/epoch_N/
│   ├── ppo_best/
│   └── trained_best_model/       # 可独立加载的完整权重和tokenizer
├── logs/
├── metrics/
├── records/
├── reports/
├── config/
├── evaluations/                 # 每次CityFlow测试独立归档
└── manifest.json
```

名称自动取算法名和微秒时间戳，不设置qwen15b_smoke等测试前缀。各次训练不覆盖；PPO按epoch保存增量权重，不按每次update保存完整模型。项目根目录不再使用results。模型、配置、必要报告和指标均归档到对应运行目录。

PPO训练使用发布版VLLMLocalGenerator、generate_offline_data、prepare_ppo_data_from_offline、evaluate_model和模型/奖励函数；vLLM生成仍先ATR后RA，保留原聊天模板。实际rollout token直接用于TRL，batch_size以query计数（每条交通记录两项）。上一轮LoRA只加载一次并保持可训练；验证分数关联真正被评估的checkpoint，最终轮也验证。vLLM阶段用同一个ppo_ft.py子进程执行，释放CUDA后再训练，没有另一份rollout算法。

## 日志与测试

SFT/PPO训练、验证、选优数据保存在metrics；教师请求费用用量与阶段耗时在logs/reports/manifest；CityFlow保存交通指标、响应文本、token与车辆观测及车辆CSV。日志不包含API密钥。本地W&B兼容记录、原TensorFlow兼容依赖均沿用已有环境，无需重装。

验收直接启动上述两个正式入口，读取主配置；my_tests仅检查输出，不调用另一套训练函数或生成专用训练配置。实际记录见 `my_tests/本轮入口测试结果.md`。

## 实现边界

发布版方法虽然命名Asynchronous_Reasoning_and_Acting，实际调用先ATR后RA；本项目直接采用该方法，不保留之前自行增加的并行缓存/故障注入路径。论文描述但发布代码未提供的完整多候选最短推理筛选也不自行补造。不能将小样本测试称为完整论文实验或充分训练的模型成绩。
