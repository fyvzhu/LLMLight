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

CityFlow正式入口统一使用发布版OpenAI兼容服务适配器，`cityflow_test.backend`固定使用`llama`；这个名称也适用于Qwen模型。先启动服务，模型参数使用与评测一致的完整模型绝对路径，端口与YAML一致，例如：

```bash
VLLM_USE_V1=0 .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /绝对路径/本次运行/models/trained_best_model \
  --host 127.0.0.1 --port 23333 --tensor-parallel-size 1 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.25 --enforce-eager
```

入口会检查服务实际提供的模型身份，避免误测其他权重。Transformers仅用于完整模型独立加载与logits检查。论文延迟实验应固定服务后端、硬件、生成参数及批量，排除预热；不能与此前Transformers本地推理计时混合。`my_tests/benchmark_inference.py`现在也调用同一服务，记录的是整批API请求往返耗时，不代表纯GPU算子耗时。

`training.enable_ppo: false` 只执行SFT并导出其验证最佳模型；true执行SFT→合并SFT最佳→PPO→导出PPO验证最佳。SFT用最小验证loss，PPO用最大验证RA reward，平分保留较早者；CityFlow测试不参与选模。

## SFT标准Checkpoint与断点续训

SFT的评估、最佳候选选择、完整训练状态保存/恢复和checkpoint轮换由现有TRL SFTTrainer及Transformers Trainer统一执行。配置使用`load_best_model_at_end=True`、`metric_for_best_model=eval_loss`、`greater_is_better=False`、`save_only_model=False`。项目回调只记录日志、检查指标、归档框架选出的Adapter；在steps策略的最终优化步骤补齐评估/保存触发标志，不另行调用评估、选优或创建checkpoint。

`sft.save_strategy`与`eval_strategy`必须相同，可选steps或epoch；steps策略的保存和评估均使用`sft.save_steps`。非整除保存间隔的最后一步也参与框架评估、选优并保存完整状态，发生在结束加载最佳模型之前。

```yaml
sft:
  save_strategy: steps
  eval_strategy: steps
  save_steps: 100
  save_total_limit: 2
```

上例仅说明参数含义，不是另一份运行配置。只修改主`config/config.yaml`。`save_total_limit`是标准checkpoint的总保留数，建议2，空间允许可设3；框架保护最佳和最近候选，不是“最佳之外再保留N个”。原生机制在limit=1时有最佳/最新保留特例，因此不把1视为严格磁盘上限。清理前会先写新checkpoint，应预留保存峰值空间。完整LoRA训练checkpoint包含Adapter、optimizer.pt、scheduler.pt、rng_state.pth、trainer_state.json，以及需要时的scaler.pt；不重复保存冻结基础模型。

每次运行的原生路径为`models/sft_checkpoints/checkpoint-N/`，不再维护attempt_N/step_N保存流程。`models/sft_best/`仅归档最佳Adapter、Tokenizer及验证元数据，不携带优化器状态。完整最佳模型的导出与PPO初始化路径保持不变；不要把best与latest混为一谈。

```bash
.venv/bin/python training/collmlight/train_collmlight.py \
  --resume-run outputs/CoLLMLight/CoLLMLight_年月日_时分秒_微秒
```

保持与该运行相同的训练配置。总入口跳过已完成阶段，SFT从同次运行最近的完整标准checkpoint恢复模型、优化器、调度器、随机数及数据进度，日志按新的attempt追加。没有checkpoint且SFT尚未开始更新时从头启动；非空旧Adapter目录、缺少状态文件或数据/配置/版本不一致时明确失败，不静默改成权重续训，不维护旧格式兼容。旧Adapter不能补回以前未保存的优化器与随机数状态。

checkpoint中已完成全部计划更新、但外层阶段尚未写完成报告时，只整理最佳归档和报告，不多执行一次更新。`manifest.json`的`stages.sft.summary`区分累计updates、此次new_updates和恢复起点resumed_step；最佳权重变化统计也明确其参照。中断后最多回退到最近一次完整保存，不恢复尚未保存的梯度或更新。同环境续训一致性需实测，不承诺任意硬件/版本间逐位一致。

本轮真实模型及恢复状态的详细验收见`my_tests/SFT断点续训测试结果.md`。Trainer返回的train_loss属于本次调用的汇总，跨中断的完整学习曲线应读取按step记录的`sft_train.jsonl`，不能直接比较不同调用的汇总loss。

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

独立采样入口`run_fts.py`和训练入口共用发布版FTS算法及同一参数转换、环境配置构造和执行函数。`sampling`配置可设置dataset、traffic_file、reward_type、reward_period、max_steps，以及ignore_threshold、long_info、feed_back、feed_back_num、debug_wollm。后几个字段未填写时沿用官方入口默认值。独立运行示例：`.venv/bin/python run_fts.py --max_sampling_steps 1`；仅显式提供的CLI参数覆盖YAML，并冻结实际采样配置。

## 输出

```text
outputs/CoLLMLight/CoLLMLight_年月日_时分秒_微秒/
├── manifest.json                # 阶段状态、attempt、各阶段简要结果、最终模型位置
├── config/
│   ├── effective.json           # 本次冻结配置；恢复时使用
│   ├── environment.json         # 环境、依赖版本、GPU信息
│   └── code.json                # 每次运行仅一份代码身份
├── data/
│   ├── traffic_samples/         # 本次采样或复制的交通输入
│   ├── teacher_data/            # 发布版六字段教师数据
│   ├── sft_data/                # train/eval、划分索引、转换拒绝原因
│   └── ppo_split.json           # PPO样本划分、来源及交叠计数
├── models/
│   ├── sft_checkpoints/checkpoint-N/ # Trainer完整训练状态；按save_total_limit轮换
│   ├── sft_best/                # 验证最佳SFT Adapter及加载、选优元数据
│   ├── sft_merged/              # PPO依赖的完整SFT基础模型
│   ├── ppo_checkpoints/epoch_N/  # 保持按epoch保存增量权重
│   ├── ppo_best/                # 最佳Adapter；不可变模型文件与对应checkpoint硬链接
│   └── trained_best_model/      # 独立完整模型；model_manifest.json内保存唯一导出验收
├── logs/
│   ├── train.log                # 全阶段及vLLM子进程共用；恢复时追加
│   ├── events.jsonl             # 阶段状态转换和耗时
│   └── teacher_requests.jsonl   # 教师请求、用量和失败信息
├── metrics/                    # SFT/PPO逐步训练、验证、选优及生成统计
├── records/sampling/            # 采样车辆原始CSV等
└── evaluations/数据集_年月日_时分秒_微秒/
    ├── config.json             # 实际评测参数、模型/数据指纹和服务身份
    ├── summary.json            # 唯一最终指标、有效动作计数和运行状态
    ├── logs/console.log
    ├── metrics/
    │   ├── traffic.jsonl        # 每步交通指标、队列及等待汇总
    │   ├── inference.jsonl      # SR/RD实际批量、整批耗时及生成参数
    │   └── tokens.jsonl         # 逐响应token、截断标记、观测车辆数；用batch_id关联
    └── records/
        ├── llm_responses.jsonl  # Prompt/响应原文仅在这里保存
        └── simulation/         # CityFlow车辆到达/离开CSV等原始记录
```

仅产生实际执行阶段所需文件。名称固定使用算法名和微秒时间戳；不设置额外实验前缀。训练不覆盖旧运行，同一模型多路网/多种子评测在同一运行的evaluations下创建不同目录。通过显式模型路径独立启动评测时会新建运行目录；通过`--run-dir`评测时不会再创建第二个训练目录。

不再生成reports目录、各阶段code_before/after副本、重复导出报告、integration_report、metrics.json、legacy_results、history_data整包转存、自动图表、未使用的dataset_info和env_rl_data。PPO指标只写标量统计，不保存token级训练数组；rollout的完整token IDs仍直接用于原TRL更新，成功提交epoch checkpoint后清理临时副本。失败时保留`tmp/`排错。仿真输入副本、导出过程文件也遵循成功后清理、失败保留，不删除正式模型或原始输入。

模型内config、Tokenizer、Adapter配置和原生checkpoint的optimizer、scheduler、RNG、trainer_state必须保留，它们属于模型格式或续训状态。PPO最佳与epoch权重使用同文件系统硬链接节约空间；可变进度与验证元数据独立复制。不要原地编辑这些模型文件；复制最佳Adapter用于外部修改时使用普通复制。

PPO训练使用发布版VLLMLocalGenerator、generate_offline_data、prepare_ppo_data_from_offline、evaluate_model和模型/奖励函数；vLLM生成仍先ATR后RA，保留原聊天模板。实际rollout token直接用于TRL，batch_size以query计数（每条交通记录两项）。上一轮LoRA只加载一次并保持可训练；验证分数关联真正被评估的checkpoint，最终轮也验证。vLLM阶段用同一个ppo_ft.py子进程执行，释放CUDA后再训练，没有另一份rollout算法。

SFT转换、PPO奖励和CityFlow共用`utils/training_rewards.py`的RA解析规则：signal必须属于四种合法信号；used_atr若提供，必须是JSON整数0或1，缺失时保留官方默认0；reason可缺省。非法输出在PPO中得到原错误奖励-1，在CityFlow中走原有兜底动作。合法输出的官方RA/SR奖励公式不变。

完整模型导出检查独立加载、tokenizer与模型架构配置、FP32合并前后logits，以及保存精度下的logits容差；随后取一条已有验证记录，复用PPO的`evaluate_model`、`*_user_prompt_wsys`和vLLM批量推理完成输出有效性验收。不再维护另一套交通政策评分函数。该验收只证明模型完整性，不重新选择best，也不提供论文ATT/AWT/AQL；后者必须来自独立CityFlow评测。

## 日志与测试

SFT/PPO训练、验证、选优数据保存在metrics；各阶段摘要在manifest中，终端输出统一追加到train.log，并标记stage、attempt及vLLM epoch。教师用量在teacher_requests；密钥不写入日志。完整模型来源与验收只在model_manifest.json保存，运行manifest仅引用其位置。

后续分析项目可使用评测summary中的ATT/AWT/AQL比较算法，使用traffic逐步指标绘制交通曲线，使用SR/RD整批耗时和实际batch_size分析延迟，使用逐次token及车辆数分析推理长度与交通复杂度；模型、路网、种子、生成配置用于区分实验条件。必须剔除或单独统计finish_reason=length的截断样本，不能把PPO的标题计数奖励当作token长度。多种子汇总、显著性检验和画图在后处理项目完成，这里不重复输出派生表格。当前服务适配器记录输出logprobs的token计数，尚未保存完整输入token用量，不能据此覆盖全部部署成本分析。

日志只记录实际执行的行为。正式延迟比较需用固定后端、预热和重复批量实验；单次普通CityFlow耗时不能替代完整基准。当前代码没有实现的异步对照、鲁棒性扰动、人工评分及额外消融，不能仅凭增加日志宣称已覆盖相应论文实验。

验收直接启动正式训练和评测入口，临时修改主配置中的普通参数后恢复，不维护另一份训练配置。测试代码与本轮结果见`my_tests/test_output_storage.py`、`my_tests/test_cli_results.py`和`my_tests/输出精简测试结果.md`；无新增依赖。

## 实现边界

发布版方法虽然命名Asynchronous_Reasoning_and_Acting，实际调用先ATR后RA；本项目直接采用该方法，不保留之前自行增加的并行缓存/故障注入路径。论文描述但发布代码未提供的完整多候选最短推理筛选也不自行补造。不能将小样本测试称为完整论文实验或充分训练的模型成绩。
