# CoLLMLight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control

> 本工作区的统一训练/导出/CityFlow 测试流程见 [TRAINING.md](TRAINING.md)。教师与学生参数保留在 `.env`，训练与测试参数见 `config/config.yaml`；最佳完整模型位于 `outputs/<run_name>/trained_best_model/`。下文原版示例作历史参考，当前入口以 TRAINING.md 为准。

Official implementation for **CoLLMLight**, accepted by ICLR 2026.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data](#data)
- [Quick Start](#quick-start)
- [Training Workflow](#training-workflow)
- [API Configuration](#api-configuration)
- [Citation](#citation)

<a id="overview"></a>
## 1. Overview

This repository provides the official implementation for CoLLMLight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control.

Network-wide optimization in traffic signal control (TSC) requires agents to cooperate across intersections. However, recent Large Language Model (LLM)-based TSC agents are designed as independent agents without inter-intersection cooperation, which limits their effectiveness at the network level. To address this gap, we propose CoLLMLight, a cooperative LLM agent framework for network-wide traffic signal control. CoLLMLight introduces a spatiotemporal-aware cooperative reasoning module to analyze interactions with neighboring agents and produce cooperative suggestions. This reasoning process is implemented within an asynchronous decision architecture to support multi-step reasoning without compromising real-time responsiveness. To further improve both cooperation effectiveness and reasoning efficiency, we propose a cost-aware cooperation optimization strategy. It first applies adaptive reasoning optimization to equip the LLM with the capability to generate concise yet effective cooperative reasoning across varied traffic conditions. Then, it refines the policy using reward signals that encourage both effective decision-making and efficient reasoning. Extensive experiments on four real-world traffic networks demonstrate that CoLLMLight significantly outperforms existing methods by enabling more effective and generalizable cooperation, while ensuring low decision latency and efficient token usage.

![Agent Framework Overview](./media/Overview.png)

<a id="requirements"></a>
## 2. Requirements

- `python>=3.9`
- `cityflow` (Requires a Linux environment; tested on Ubuntu)
- `tensorflow-cpu==2.8.0` or `tensorflow-gpu==2.8.0`
- `torch==2.2.2`
- `transformers==4.48.2`
- `trl==0.9.2`
- `vllm`
- `lmdeploy`

You can install the required Python packages using:

```bash
pip install -r requirements.txt
```

<a id="data"></a>
## 3. Data

The repository follows the CityFlow data layout. Each traffic network contains a roadnet file and one or more traffic flow files:

```text
data/
├── Hangzhou/4_4/
├── Jinan/3_4/
├── NewYork/28_7/
├── Synthetic/4_4/
└── FinetuneData/
```

For example, the New York benchmark uses:

- Roadnet: `data/NewYork/28_7/roadnet_28_7.json`
- Traffic flow: `data/NewYork/28_7/anon_28_7_newyork_real_double.json`

<a id="quick-start"></a>
## 4. Quick Start

The updated local inference entry loads the full model configured in `config/config.yaml` (`cityflow_test.model_path`). By default it uses `outputs/<run_name>/trained_best_model/`. Run from the project directory with the existing uv environment:

```bash
.venv/bin/python run_CoLLMlight.py --config config/config.yaml
```

The tested backend is local Transformers; no inference server is required. Original HTTP client classes remain available, but remote serving is outside this workspace's integration test.

<a id="training-workflow"></a>
## 5. Training Workflow

See [TRAINING.md](TRAINING.md) for configuration, data sampling, real teacher generation, TRL LoRA-SFT, SFT merge, LoRA-PPO, checkpoint recovery, best full-model export and CityFlow testing.

```bash
.venv/bin/python my_tests/run_real_stages.py teacher
.venv/bin/python my_tests/run_real_stages.py prepare
.venv/bin/python my_tests/run_real_stages.py sft
.venv/bin/python my_tests/run_real_stages.py merge
.venv/bin/python my_tests/run_real_stages.py ppo
.venv/bin/python export_best_model.py
.venv/bin/python my_tests/run_cityflow_model_smoke.py
```

Run real CityFlow sampling first if `data/FinetuneData/SynTrain_sample_all.json` is missing. Teacher generation uses paid API calls and resumes completed examples. Do not blindly rerun training in an existing experiment directory; choose a new `experiment.run_name`.

<a id="api-configuration"></a>
## 6. Configuration

- `.env` / `.env.example`: teacher and student model settings; the real API key belongs only in `.env` or the process environment.
- `config/config.yaml`: data tasks, LoRA-SFT, LoRA-PPO, export and CityFlow test settings. The obsolete `ppo_config.yaml` is no longer used.
- `llm_models/`: downloaded foundation models; `models/` remains the traffic agent source directory.
- `my_tests/`: added tests and execution logs; `tests/` retains the original regression tests.
- `outputs/<run_name>/trained_best_model/`: validation-checked, standalone full inference model.

All relative configuration paths resolve against this project directory. See [the second-round test report](my_tests/第二轮测试结果.md) for actual validation scope and results.

<a id="citation"></a>
## 7. Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@inproceedings{yuan2026collmlight,
  title={Co{LLML}ight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control},
  author={Yuan, Zirui and Lai, Siqi and Liu, Hao},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=KeJqoEVOeY}
}
```
