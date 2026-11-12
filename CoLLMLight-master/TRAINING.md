# CoLLMLight 训练与独立评测

2026-11-12 更新。当前默认配置只用于工程调通，不是论文全量参数；完整实测记录见 `my_tests/第三轮测试结果.md`。

## 入口和目录

- **总训练入口**：`training/collmlight/train_collmlight.py`。
- **CityFlow 评测入口**：根目录 `run_CoLLMlight.py`，只评测，不运行 SFT/PPO。
- 原七份主流程训练脚本迁入 `training/collmlight/`，名字保留；历史 `reasoning_refinement.py` 在其 `legacy/` 中，不自动执行。
- `framework/CoLLMlight.py`、`framework/FTSample.py` 保留控制和采样实现；对比算法继续使用各自根目录 `run_xx.py`，`models/` 保留智能体。

所有命令从 `CoLLMLight-master/` 执行，使用已有 `.venv/bin/python`。不要从系统 Python 或另一套环境启动。

```bash
.venv/bin/python -m training.collmlight.train_collmlight --help
.venv/bin/python -m training.collmlight.train_collmlight --dry-run
.venv/bin/python -m training.collmlight.train_collmlight --config config/config.yaml
```

直接执行 `.venv/bin/python training/collmlight/train_collmlight.py --config config/config.yaml` 也受支持。旧的 `python run_sft.py` / `python ppo_ft.py` 等根目录命令已迁移，不再保留两份实现；独立阶段使用 `python -m training.collmlight.<原模块名> --config ... --run-dir <已初始化实验>`。

## 配置：仅 SFT 或两阶段

教师/学生连接、模型和凭证继续在 `.env`，不写入 YAML 或上传日志。`config/config.yaml` 是任务配置来源：

```yaml
training:
  enable_ppo: true              # false：仅 SFT；true：SFT + PPO
  data_preparation: reuse       # 默认不产生教师费用
  sft_source_run: null
export:
  source: auto
  output_dir: null
  validate: true
  auto_export: true
```

- `reuse`：读取 `teacher_data.output_path` 指定的已有真实教师数据，复制到本次实验后转换；缺数据时报错，不自行付费调用。
- `generate_teacher`：从 `teacher_data.source_path` 生成指定条数教师数据，按记录保存进度。
- `sample_and_generate`：先进行小规模 CityFlow 采样，再调用教师；公共输入文件不被覆盖。
- `teacher_data.max_samples` 与 `ppo.max_samples` 的 **0 显式表示全部有效记录**；默认分别5、4，不会自动全量。

两模式都验证 SFT checkpoint，选择最低 `eval_loss` 的 `sft_best`。仅 SFT 模式将它直接合并为完整最终模型；两阶段模式以它的完整合并模型为基础做 PPO，再按原版验证平均 RA reward 选择 `ppo_best`。最后一轮 PPO 也会被验证；平分保留较早者。

SFT 的最低 loss 和 PPO 的最高 RA reward 是不同选模标准，均不等于 CityFlow ATT 已最优；测试集不用于选模型。PPO 失败不会自动回退到 SFT 并报告两阶段成功。

## 当前小规模参数与原版恢复边界

reuse不会按teacher_data.max_samples截断已有文件；该上限属于生成阶段。当前指定输入文件确有5条，启动预检会打印实际复用条数。

当前保持：教师5条（已有文件包含5条）、SFT最多2次更新、每步保存验证，PPO最多4条有效交通记录、2个外层 epoch、batch=4 query、mini_batch=2、累积1、ppo_epochs=1、生成上限1024、短仿真60秒。

已经恢复原版：每 epoch vLLM 集中 ATR→集中 RA→释放进程→Transformers/TRL 更新，PPO NF4 4bit双重量化、BF16计算、r=8/alpha=16/dropout=0、PEFT默认q/v目标层、奖励白化、按epoch保存。保留批量实际 query 计数、正确加载上一轮 Adapter、模板/token一致及选模权重错位等必要修复。

**不会因代码还原就默认运行100轮或全量。** `config/ppo_original_reference.yaml` 只记录原仓库100轮、batch32等参数，不会被训练入口读取，也不是完整运行配置。扩容后再将选定参数合并进实际配置。SFT 原版没有完整实现，其设置属于本项目补齐模块，不冒充作者参数。

## 实验输出与比较

新实验目录为：

```text
outputs/CoLLMLight/<实验名称_年月日_时分秒_微秒>/
├── manifest.json
├── config/effective.json        # 固定训练配置，校验摘要
├── data/                        # 教师数据、SFT转换、PPO划分和原始rollout
├── models/
│   ├── sft_checkpoints/         # attempt/step或epoch对应的保存点
│   ├── sft_best/
│   ├── sft_merged/              # 两阶段需要
│   ├── ppo_checkpoints/epoch_N/ # 两阶段需要；不逐update保存优化器
│   ├── ppo_best/                # 两阶段需要
│   └── trained_best_model/      # 两种模式都有：完整权重、tokenizer和验证清单
├── logs/
├── metrics/                     # 逐步JSONL；不只有终端输出
├── records/
├── evaluations/                # 每次评测独立时间戳目录
└── reports/
```

新模型使用 `sft_best/`，历史实验的 `sft_adapter/` 不重命名。失败尝试保留日志，显式恢复同一配置：

```bash
.venv/bin/python -m training.collmlight.train_collmlight --config <原配置.yaml> --resume-run <原实验目录>
```

PPO先支持已完成epoch边界恢复；原版各epoch重建训练器的规则保留，不宣称从未保存的中间update精确恢复。不能修改模式后原地恢复，也不能把旧 `update_N` checkpoint 静默当新基线继续。

公平比较建议：先用 `enable_ppo: false` 训练；再新建配置设 true，并把 `training.sft_source_run` 指向这份完成的 SFT 实验。新实验复用已验证的同一 SFT Adapter，独立训练 PPO，保留父实验来源和摘要。不要用两次独立 SFT 的随机差异来解释 PPO 增益。所有完整模型文件应视为只读；跨实验复用可能使用相同文件系统的权重硬链接，删除一个目录不会删除另一个目录的模型，不能原地修改共享权重文件。

## 独立运行 CityFlow

```bash
.venv/bin/python run_CoLLMlight.py --run-dir <本次训练实验目录>
```

或者在 YAML 的 `cityflow_test.model_path` 填写某份完整模型，省略 `--run-dir`，创建独立评测实验。不要传 Adapter 目录。两个显式来源冲突会报错，不按“最新目录”猜测或退回基础模型。

支持本地 `transformers` 及原兼容 `llama` 服务端协议。服务端必须实际加载指定完整权重，入口先查询 `/v1/models` 核对模型身份。例如在独立终端启动已安装的 vLLM：

```bash
USE_TF=0 VLLM_USE_V1=0 .venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model <完整模型绝对路径> --host 127.0.0.1 --port 23333 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.22 --enforce-eager

.venv/bin/python run_CoLLMlight.py --run-dir <实验目录> --backend llama --port 23333
```

`--model_type llama`、`--model_path`、`--traffic_file`、`--new_max_tokens`、`--disable_psr`、`--num_rounds` 保留常用命令兼容。当前配置使用单GPU；不在这里提供多卡训练调度。

每次评测保存console日志、完整模型/数据身份、车辆CSV、状态/动作、逐步交通指标、逐次SR/RD延迟和token等。原版 `avg_atr_time` 是摊销值；画延迟图应使用 `batch_wall_time_s`。SR奖励的标题数量仍是原公式，不能当生成token数。指标定义沿用原代码，详情见数据字典和报告。

## 可选实验与数据导出

- `--sr-mode original` 默认保持原控制流程；`sync_sr`、`without_sr`、`async_sr`、`stale_sr`、`communication_failure` 显式选择新增实验模式。异步首步无缓存时用空SR，结束时等待未完成任务以完整落盘；结尾排空耗时不当成RD决策延迟。
- 故障抽样按每个agent决策，独立seed；通信失败最多选一个已有邻居，明确标记不可用。缓存空缺/实际age/故障事件均保留，不能把短程全cache-miss实验当成熟异步效果。
- `python -m training.collmlight.run_ablation --variant <full|without_pr|without_ar|without_both> --dry-run` 查看独立消融入口。w/o AR采用“原基础模型直接PPO”的明确协议；并非作者未公开设置的复原。w/o Both是未训练基础模型引用，不标成验证最佳微调模型。

```bash
.venv/bin/python -m training.collmlight.benchmark_inference --run-dir <实验目录> --repeats 2
.venv/bin/python -m training.collmlight.export_metrics --run-dir <实验目录>
```

基准输出SR/RD、b=1/5/10的逐次整批时间；短预算64-token测试只验证采集管线，不作为论文性能数据。正式测量应固定提示集合、硬件、后端、生成协议并独占GPU。CSV只导出本实验的字段，不自动拼跨算法表格。

## 环境、测试和限制

主训练栈已实测 Transformers4.51.3、TRL0.9.2、PEFT0.14.0、vLLM0.9.1、bitsandbytes0.46.1、Torch2.7.0+cu126。新增/调整历史见工作区根目录 `new_requirements.txt`，它是变更记录，不是可直接整体安装的锁文件。

旧RL使用原tensorflow-cpu2.8.0。主环境protobuf7.36.2供vLLM等使用；仅在 `.venv/compat/tensorflow28` 隔离安装 protobuf3.20.3，`utils/config.py` 在独立RL进程优先加载它，不替换主环境。若此隔离目录缺失，应按记录安装到该target，不能全局降级protobuf。W&B未安装时使用本地追踪兼容层，不要求网络账号，不影响算法更新。

```bash
.venv/bin/python -m unittest discover -s my_tests -v
.venv/bin/python -m unittest discover -s tests -v
```

真实教师/训练/恢复/CityFlow专项脚本及证据位于 `my_tests/stage3/`；新一轮实测必须重新建立自己的run_index，不能把旧证据当新训练。旧实验若被手动移除，依赖它的旧Adapter/符号链接不会自动变成有效模型，不能改指向另一份不相同的权重。

本轮只完成小规模工程和实验机制验证；未运行100轮、全数据、8B训练或完整论文多算法长时长对比。资源不足时入口在启动前报错，不删除历史实验腾空间。
