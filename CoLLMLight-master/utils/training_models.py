"""Shared student loading and exact chat tokenization (imported only by GPU stages)."""
import importlib.metadata
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from .training_config import atomic_json


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_for(cfg):
    choice = cfg['STUDENT_DTYPE']
    if choice == 'auto':
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if choice not in ('float32', 'float16', 'bfloat16'):
        raise ValueError('STUDENT_DTYPE must be auto/float32/float16/bfloat16')
    if choice == 'bfloat16' and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bfloat16')
    return getattr(torch, choice)


def tokenizer_for(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_student(cfg, path, stage, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('This training stage requires CUDA')
    bits = cfg[f'{stage}_QUANTIZATION_BITS']
    if bits not in (0, 4):
        raise ValueError('Quantization must be 0 or 4 bits')
    kwargs = dict(local_files_only=True, torch_dtype=dtype_for(cfg), attn_implementation='sdpa')
    if bits:
        if device != 'cuda':
            raise ValueError('4bit training requires CUDA')
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_for(cfg), bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True),
            device_map={'': 0})
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if cfg['STUDENT_MAX_LENGTH'] > model.config.max_position_embeddings:
        raise ValueError('STUDENT_MAX_LENGTH exceeds model context')
    if not bits:
        model.to(device)
    model.config.use_cache = False
    return model


def lora_for(cfg, stage):
    targets = cfg[f'{stage}_LORA_TARGET']
    return LoraConfig(task_type='CAUSAL_LM', r=cfg[f'{stage}_LORA_RANK'],
                      lora_alpha=cfg[f'{stage}_LORA_ALPHA'],
                      lora_dropout=cfg[f'{stage}_LORA_DROPOUT'],
                      target_modules=None if targets == 'auto' else (targets if targets == 'all-linear' else targets.split(',')))


def query_ids(tokenizer, system, prompt, max_length, reserve):
    ids = tokenizer.apply_chat_template([{'role': 'system', 'content': system},
                                        {'role': 'user', 'content': prompt}],
                                       tokenize=True, add_generation_prompt=True)
    if len(ids) + reserve > max_length:
        raise ValueError(f'Prompt ({len(ids)}) + generation budget ({reserve}) exceeds {max_length}; increase context or reduce output budget')
    return ids


def versions(cfg):
    report = {}
    for name in ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets', 'python-dotenv'):
        report[name] = importlib.metadata.version(name)
    report['cuda'] = torch.version.cuda
    report['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    from .training_config import read_json
    path=cfg.out/'config/environment.json'
    environment=read_json(path)
    environment['packages'].update({k:v for k,v in report.items() if k not in ('cuda','gpu')})
    environment.update(cuda=report['cuda'],gpu=report['gpu'],gpu_count=torch.cuda.device_count())
    atomic_json(path,environment)


def trainable_snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def weight_change(model, before):
    changed = [n for n, p in model.named_parameters() if n in before and not torch.equal(p.detach().cpu(), before[n])]
    if not changed:
        raise RuntimeError('No trainable weights changed')
    if not any('lora_' in n for n in changed):
        raise RuntimeError('No LoRA weights changed')
    return changed


def frozen_fingerprint(model):
    """Hash frozen parameter storage in bounded chunks, without a full model copy."""
    import hashlib
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:continue
        h.update(name.encode())
        flat=p.detach().reshape(-1)
        for chunk in flat.split(4*1024*1024):
            h.update(chunk.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
