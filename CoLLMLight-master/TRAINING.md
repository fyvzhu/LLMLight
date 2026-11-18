# CoLLMLight 训练与评测

本项目采用 ICLR 2026 发布代码的 ATR/RA 教师生成、PPO 及 CityFlow 控制流程。补充的部分是 LoRA-SFT、阶段衔接、统一配置、日志和完整最佳模型导出；原代码中的批量计数、Adapter 恢复和验证选模错误保留必要修复。没有旧版教师格式兼容层，没有额外实验训练入口，也不输出基线比对 JSON。

## 环境与配置

使用现有 `CoLLMLight-master/.venv`。教师和学生模型连接设置在 `.env`；训练与评测参数统一在 `config/config.yaml`。只修改这一份主配置即可调整数据量和训练轮次，不需要建立“调通专用”配置。

当前学生配置示例使用 `STUDENT_MAX_LENGTH=12288`（在`.env`中设置），覆盖本次Qwen全量SFT样本及PPO提示词加生成预算。该值是完整聊天序列的上下文预算，不是输出token上限；相较原版8192属于明确的配置差异，仍须满足实际基础模型的上下文上限和显存条件。已冻结为8192的失败训练目录不直接改配置续训，应复用现有数据新建训练运行。

输出根目录为顶层 `output_root`，训练随机种子在 `training.random_seed`。算法名称固定为CoLLMLight，不在YAML中配置，也没有experiment分组或实验模式开关。教师数据抽样和CityFlow评测分别使用其各自的random_seed/seed。

```yaml
output_root: ./outputs
training:
  random_seed: 42
  data_preparation: sample_and_generate
```

当前主配置采用完整4×4仿真采样（120个30秒决策步）、最多1000条合格教师记录、SFT完整1个epoch、PPO完整100个epoch。采样断点默认每5个完整决策步保存一次。小规模检查可直接调整同一YAML中的数据和训练上限，不需要另一套配置。

## 三个训练入口与独立评测

```bash
# 完整训练；默认读取config/config.yaml
.venv/bin/python training/collmlight/train_collmlight.py

# 仅 SFT，包含验证、最佳 Adapter 与完整模型导出
.venv/bin/python training/collmlight/train_sft.py

# 仅 PPO，明确指定 SFT 来源；来源只需 SFT 阶段完成
.venv/bin/python training/collmlight/train_ppo.py --sft-run /绝对路径/SFT运行目录

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

三个入口共用 `pipeline.py`；由入口决定阶段，配置中不再保留训练阶段开关或 SFT 来源字段。`train_ppo.py` 使用 `--sft-run` 指定来源，新建独立运行目录；来源可以是 SFT 完成但 PPO 失败的整流程目录。SFT用最小验证loss，PPO用最大验证RA reward，平分保留较早者；CityFlow测试不参与选模。

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

保持与该运行相同的训练配置，并使用最初创建该运行的同一个入口。整流程使用上例；仅 SFT 将文件名改为 `train_sft.py`，仅 PPO 改为 `train_ppo.py`，恢复时不再传 `--sft-run`。入口跳过已完成阶段，SFT从同次运行最近的完整标准checkpoint恢复模型、优化器、调度器、随机数及数据进度，日志按新的attempt追加。没有checkpoint且SFT尚未开始更新时从头启动；非空旧Adapter目录、缺少状态文件或数据/配置/版本不一致时明确失败，不静默改成权重续训，不维护旧格式兼容。旧Adapter不能补回以前未保存的优化器与随机数状态。

checkpoint中已完成全部计划更新、但外层阶段尚未写完成报告时，只整理最佳归档和报告，不多执行一次更新。`manifest.json`的`stages.sft.summary`区分累计updates、此次new_updates和恢复起点resumed_step；最佳权重变化统计也明确其参照。中断后最多回退到最近一次完整保存，不恢复尚未保存的梯度或更新。同环境续训一致性需实测，不承诺任意硬件/版本间逐位一致。

Trainer返回的train_loss属于本次调用的汇总，跨中断的完整学习曲线应读取按step记录的`sft_train.jsonl`，不能直接比较不同调用的汇总loss。

## PPO 生成与轮内断点恢复

```yaml
ppo:
  rollout_batch_size: 16
  save_steps: 5
  save_total_limit: 2
```

这些字段已经位于唯一主配置中。生成每完成一个小批就提交文本结果、实际token统计、模型/数据/参数身份及进度；例如ATR已全部保存、RA已保存400条，恢复时不会重做ATR或这400条RA。尚未提交的小批需重做，不恢复GPU内部生成到一半的token或KV cache。

当前轮缓存位于 `tmp/ppo_rollouts/epoch_N/`，四个阶段为 train_atr、train_ra、eval_atr、eval_ra。缓存只用于当前轮恢复，不永久保存全部历史rollout。不同模型、数据、代码或生成参数不能复用同一缓存。

在完整 `trainer.step()` 返回后，按 `ppo.save_steps` 保存轮内状态，第一步前和最后一个完整更新也保存。原生Accelerate状态包含优化器、存在时的调度器、随机数及梯度累积计数；保存钩子仅写LoRA与价值头，另记录PPO控制器/统计、epoch、下一条交通记录位置和rollout身份，不重复保存冻结基础权重。断点位于 `checkpoints/ppo_resume/epoch_N_update_U/`，成功提交新断点后才轮换，默认保留最近2份。

恢复命令始终指定原运行目录，并使用相同入口：

```bash
# 整流程运行恢复
.venv/bin/python training/collmlight/train_collmlight.py --resume-run /绝对路径/原运行目录
# PPO-only 运行恢复
.venv/bin/python training/collmlight/train_ppo.py --resume-run /绝对路径/原PPO运行目录
```

已完成的epoch跳过；未完成epoch从最近成功保存的完整更新边界继续，复用同轮rollout。未保存的更新需重做。轮内恢复优化器和价值头，不改变官方跨epoch重新初始化优化器/价值头的行为；完整epoch的Adapter仍保存到 `models/ppo_checkpoints/epoch_N/`。

结构化PPO指标记录run、attempt、epoch、global_update及rollout身份。断点同时提交指标字节位置；恢复时回滚未被模型断点采用的指标尾部，而不是只保留最后一次attempt。已完成epoch和SFT指标保留，`logs/train.log`继续保存分attempt的原始历史。不要将控制台进度条当作正式实验指标源。

旧目录的SFT成果仍可通过 `train_ppo.py --sft-run <旧目录>` 复用，包括SFT成功但PPO失败的目录。旧版本未保存的生成结果、优化器和RNG不能事后补回；本次不修改历史目录配置、不提供旧开关或旧断点格式兼容。配置schema改变的历史运行请作为SFT来源创建新的PPO运行，而不是在历史目录强行切换配置。

保留的测试入口与验收结论见 [测试说明](my_tests/README.md)。默认 BF16/SDPA 更新不保证逐位确定；状态恢复与确定性算子下的对照已分别验证。历史测试模型和日志已经清理。

## 数据准备模式和输入

| data_preparation | inputs.traffic_samples | inputs.teacher_data |
| --- | --- | --- |
| sample_and_generate | 忽略，自动采样 | 忽略，自动生成 |
| generate_teacher | 必填 | 忽略，自动生成 |
| reuse，整流程 | 必填，作为PPO场景输入 | 必填 |
| 仅PPO入口 | 必填 | 不使用；SFT来源由 --sft-run 指定 |
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

## 采样断点恢复

`sampling.checkpoint_interval_steps: 5`表示每完成5个采样决策步保存一次；最后一步即使不足间隔也保存。每步先完成所有路口候选试验、执行真实动作并更新历史及累计指标，再提交快照。保持官方FTS决策、前瞻和奖励计算不变。

每次运行只保留`checkpoints/sampling/latest.ckpt`一份有效断点。它打包CityFlow原生快照、Python环境与智能体历史、已完成样本、循环进度、随机数状态以及仿真日志字节位置。新文件校验并刷盘后原子替换旧文件；写入期间短暂需要额外空间。失败或强制结束可能留下同目录的`.writing`临时目录，下一次保存会清理，不会把它当成有效断点。正式采样数据及完成状态提交成功后清理恢复文件。

```bash
# 独立采样；输出Sampling run目录
.venv/bin/python run_fts.py

# 独立采样中断后恢复，指定同一次运行目录
.venv/bin/python run_fts.py --resume-run outputs/CoLLMLight/CoLLMLight_年月日_时分秒_微秒

# 总训练入口中断后恢复，采样阶段也使用同一套恢复逻辑
.venv/bin/python training/collmlight/train_collmlight.py --resume-run outputs/CoLLMLight/CoLLMLight_年月日_时分秒_微秒
```

独立采样运行用run_fts恢复，总训练运行用train_collmlight恢复。恢复使用原运行冻结配置，不创建新的时间戳目录；总训练入口要求主配置与原运行一致。采样会核对代码、Python/NumPy/CityFlow、配置以及路网/车流文件指纹，不做旧格式兼容或静默从头重跑。仅有旧JSON而没有完整断点的运行不能精确恢复。

例如第7步试验途中中断，最近断点是完成第5步，恢复后从第6步继续；未保存的第6步需要重做。恢复时JSON和仿真日志回退到断点对应状态，终端诊断日志继续追加，保留中断记录。断点内Python状态只允许加载自己生成、可信且未被他人修改的本地文件。

断点机制不缩短采样本身的计算时间，也不保证恢复尚未保存的进度。仍建议在tmux中运行。此前已使用真实CityFlow进行不同进程的连续/中断恢复对照；当前保留`my_tests/test_sampling_checkpoint.py`进行断点提交与恢复边界回归检查。

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
├── checkpoints/ppo_resume/     # 当前轮最近完整更新状态；成功提交epoch后清理
├── checkpoints/sampling/latest.ckpt # 采样期间唯一恢复点，成功提交后清理
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

不再生成reports目录、各阶段code_before/after副本、重复导出报告、integration_report、metrics.json、legacy_results、history_data整包转存、自动图表、未使用的dataset_info和env_rl_data。PPO指标只写标量统计，不保存token级训练数组；rollout文本按官方方式重新编码，query 最长8192、response 最长2048，分别截断；成功提交epoch checkpoint后清理当前轮生成缓存和轮内断点。失败时保留`tmp/`排错。仿真输入副本、导出过程文件也遵循成功后清理、失败保留，不删除正式模型或原始输入。

模型内config、Tokenizer、Adapter配置和原生checkpoint的optimizer、scheduler、RNG、trainer_state必须保留，它们属于模型格式或续训状态。PPO最佳与epoch权重使用同文件系统硬链接节约空间；可变进度与验证元数据独立复制。不要原地编辑这些模型文件；复制最佳Adapter用于外部修改时使用普通复制。

PPO使用发布版提示词、生成、奖励与验证协议；生成结果组装和缓存共用一套实现，仍按整轮先ATR后RA、完成生成后再更新，保留原聊天模板。query和response按官方规则分别重新编码及截断，不保留生成token IDs训练路径；batch_size以query计数（每条交通记录两项）。正式更新保留kbit准备函数，修正实际前向模式并使用BF16 autocast。上一轮LoRA只加载一次并保持可训练；验证分数关联真正被评估的checkpoint，最终轮也验证。vLLM阶段用同一个ppo_ft.py子进程执行，释放CUDA后再训练，没有另一份rollout算法。

SFT转换、PPO奖励和CityFlow共用`utils/training_rewards.py`的RA解析规则：signal必须属于四种合法信号；used_atr若提供，必须是JSON整数0或1，缺失时保留官方默认0；reason可缺省。非法输出在PPO中得到原错误奖励-1，在CityFlow中走原有兜底动作。合法输出的官方RA/SR奖励公式不变。

完整模型导出检查独立加载、tokenizer与模型架构配置、FP32合并前后logits，以及保存精度下的logits容差；随后取一条已有验证记录，复用PPO的`evaluate_model`、`*_user_prompt_wsys`和vLLM批量推理完成输出有效性验收。不再维护另一套交通政策评分函数。该验收只证明模型完整性，不重新选择best，也不提供论文ATT/AWT/AQL；后者必须来自独立CityFlow评测。

## 日志与测试

SFT/PPO训练、验证、选优数据保存在metrics；各阶段摘要在manifest中，终端输出统一追加到train.log，并标记stage、attempt及vLLM epoch。 统一训练入口同时实时转发子进程输出到终端，包含嵌套的vLLM生成和导出验收，不在每层重复写日志。采样显示仿真步、路口、候选信号、已采样数和ATT/AWT/AQL；教师显示ATR/RA任务、完成数、请求耗时、重试次数及累计token；SFT沿用Trainer原生进度和指标；PPO显示epoch、更新进度、reward、policy/value loss及KL。逐秒time/Step time明细仅留在train.log，避免淹没终端。子进程连续30秒没有输出时显示存活提示和已用时间，不伪造完成百分比；Ctrl+C会停止本次启动的子进程组并记录阶段中断状态。教师用量在teacher_requests；密钥不写入日志。完整模型来源与验收只在model_manifest.json保存，运行manifest仅引用其位置。

后续分析项目可使用评测summary中的ATT/AWT/AQL比较算法，使用traffic逐步指标绘制交通曲线，使用SR/RD整批耗时和实际batch_size分析延迟，使用逐次token及车辆数分析推理长度与交通复杂度；模型、路网、种子、生成配置用于区分实验条件。必须剔除或单独统计finish_reason=length的截断样本，不能把PPO的标题计数奖励当作token长度。多种子汇总、显著性检验和画图在后处理项目完成，这里不重复输出派生表格。当前服务适配器记录输出logprobs的token计数，尚未保存完整输入token用量，不能据此覆盖全部部署成本分析。

日志只记录实际执行的行为。正式延迟比较需用固定后端、预热和重复批量实验；单次普通CityFlow耗时不能替代完整基准。当前代码没有实现的异步对照、鲁棒性扰动、人工评分及额外消融，不能仅凭增加日志宣称已覆盖相应论文实验。

保留的真实模型集成测试直接启动正式训练入口，在my_tests中创建临时小样本配置，不修改主配置，不维护另一套训练实现。测试用途与执行命令见`my_tests/README.md`；输出边界回归检查为`my_tests/test_output_storage.py`。历史一次性脚本、重复报告和测试输出已清理；无新增依赖。

## 实现边界

发布版方法虽然命名Asynchronous_Reasoning_and_Acting，实际调用先ATR后RA；本项目直接采用该方法，不保留之前自行增加的并行缓存/故障注入路径。论文描述但发布代码未提供的完整多候选最短推理筛选也不自行补造。不能将小样本测试称为完整论文实验或充分训练的模型成绩。
