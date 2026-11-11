"""Merge a trained SFT adapter into a standalone Transformers model."""
from utils.training_config import TrainingConfig, atomic_json, config_args


def merge(cfg):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from utils.training_models import tokenizer_for
    path = cfg.out / 'sft_merged'
    if (path / 'config.json').exists():
        raise FileExistsError('Merged model already exists; use a new RUN_NAME to avoid stale weights')
    tokenizer = tokenizer_for(cfg.out / 'sft_adapter')
    base = AutoModelForCausalLM.from_pretrained(cfg.path('STUDENT_MODEL_PATH'),
        torch_dtype=torch.float32, local_files_only=True).to(cfg['MERGE_DEVICE'])
    model = PeftModel.from_pretrained(base, cfg.out / 'sft_adapter').eval()
    inputs = tokenizer('Return a traffic signal: ETWT', return_tensors='pt').to(cfg['MERGE_DEVICE'])
    with torch.no_grad():
        before = model(**inputs).logits[:, -1].float().cpu()
    model = model.merge_and_unload(safe_merge=True).eval()
    with torch.no_grad():
        after = model(**inputs).logits[:, -1].float().cpu()
    error = (before - after).abs().max().item()
    if not torch.allclose(before, after, atol=5e-4, rtol=5e-4):
        raise RuntimeError(f'SFT merge logits mismatch: {error}')
    # Save compact BF16/FP16-compatible weights; compare merge itself in FP32 above.
    model.to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    atomic_json(cfg.out / 'merge_report.json', {'max_logit_difference_fp32': error,
        'source_model': str(cfg.path('STUDENT_MODEL_PATH')), 'sft_adapter': str(cfg.out / 'sft_adapter'),
        'merged_model': str(path), 'saved_dtype': 'bfloat16'})
    print('SFT merged:', path, 'max logit difference:', error)


if __name__ == '__main__':
    args = config_args(__doc__).parse_args()
    merge(TrainingConfig(args.env_file, config_file=args.config_file))
