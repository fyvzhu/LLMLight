# CoLLMLight: Cooperative Large Language Model Agents for Network-Wide Traffic Signal Control

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

To run inference with a trained model, follow these steps.

**Step 1: Deploy the LLM Inference Server**

Deploy your chosen LLM with an OpenAI-compatible local inference service such as `lmdeploy` or `vllm`. The `model_path` should be a local path or Hugging Face-compatible model identifier.

```bash
# Example using lmdeploy
lmdeploy serve api_server /path/to/your/llm --tp <num_gpus>
```

**Step 2: Run the Simulation**

Set the complete model path in `config/config.yaml`, or select a training run:

```bash
.venv/bin/python run_CoLLMlight.py --run-dir outputs/CoLLMLight/CoLLMLight_<timestamp>
```

The configured `transformers` backend can also load a local complete model directly, without a separate inference server.

<a id="training-workflow"></a>
## 5. Training Workflow

Use the existing project environment and the single main configuration `config/config.yaml`:

```bash
.venv/bin/python training/collmlight/train_collmlight.py
```

`training.data_preparation` selects simulation sampling plus teacher generation, generation from an existing traffic sample file, or existing teacher data. `training.enable_ppo` selects SFT-only or SFT followed by PPO. Teacher generation uses the published ATR/RA functions and six-field data format. The added SFT stage connects to the published PPO implementation with the required batch, adapter-loading and validation fixes.

Each run stores its model, logs and metrics under `outputs/CoLLMLight/CoLLMLight_<timestamp>/`. The final `models/trained_best_model/` is a complete independently loadable model. See [TRAINING.md](TRAINING.md) for input activation rules and commands. Teacher/student credentials remain in `.env`.

<a id="api-configuration"></a>
## 6. API Configuration

`utils/LLMs.py` supports OpenAI-compatible teacher services. This project loads the configured teacher parameters and secret variable name from `.env`; see `.env.example`. The original upstream API conventions are:

- `OPENAI_API_KEY`: API key for the chat completion service.
- `OPENAI_BASE_URL`: Base URL for an OpenAI-compatible API. Defaults to `https://api.openai.com/v1`.
- `OPENAI_API_URL`: Full chat completions URL. If set, this takes priority over `OPENAI_BASE_URL`.

Examples:

```bash
# Official OpenAI endpoint
export OPENAI_API_KEY="your_api_key"

# Custom OpenAI-compatible endpoint
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://your-endpoint.example.com/v1"
```

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
