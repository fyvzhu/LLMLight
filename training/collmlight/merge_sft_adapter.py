# Direct script execution uses the same project root as module execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
"""Merge a trained SFT adapter into a standalone Transformers model."""
from utils.training_config import TrainingConfig, atomic_json, config_args


from utils.experiment import exclusive_stage, stage_summary

@exclusive_stage
def merge(cfg):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from utils.training_models import tokenizer_for
    path = cfg.artifact('sft_merged')
    if (path / 'config.json').exists():
        raise FileExistsError('Merged model already exists; use a new RUN_NAME to avoid stale weights')
    if cfg.sft_source:
        import shutil
        from utils.training_config import read_json
        from utils.experiment import link_model_file
        from training.collmlight.export_best_model import file_hash, weights_fingerprint
        best_hash=file_hash(cfg.artifact('sft_adapter')/'adapter_model.safetensors')
        state=read_json(cfg.sft_source/'manifest.json')
        source=cfg.sft_source/'models/sft_merged'
        reusable=(source/'config.json').is_file() and state.get('stages',{}).get('merge',{}).get('state')=='completed'
        if not reusable:
            source=cfg.sft_source/'models/trained_best_model'
            meta=read_json(source/'model_manifest.json') if (source/'model_manifest.json').exists() else {}
            reusable=meta.get('selected_stage')=='sft' and meta.get('source_adapter_sha256')==best_hash
            if reusable and weights_fingerprint(source)!=meta['exported_weights']:raise ValueError('Source full SFT weights changed')
        if reusable:
            shutil.copytree(source,path,copy_function=link_model_file)
            stage_summary(cfg,'merge',{'merged_model':str(path),'reused_sft_full_model':str(source),
                'source_adapter_sha256':best_hash,'weights':weights_fingerprint(path),'storage':'immutable weight hardlinks'})
            print('Verified SFT full weights reused:',path)
            return
    tokenizer = tokenizer_for(cfg.artifact('sft_adapter'))
    base = AutoModelForCausalLM.from_pretrained(cfg.path('STUDENT_MODEL_PATH'),
        torch_dtype=torch.float32, local_files_only=True).to(cfg['MERGE_DEVICE'])
    model = PeftModel.from_pretrained(base, cfg.artifact('sft_adapter')).eval()
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
    stage_summary(cfg,'merge', {'max_logit_difference_fp32': error,
        'source_model': str(cfg.path('STUDENT_MODEL_PATH')), 'sft_adapter': str(cfg.artifact('sft_adapter')),
        'merged_model': str(path), 'saved_dtype': 'bfloat16'})
    print('SFT merged:', path, 'max logit difference:', error)


if __name__ == '__main__':
    args = config_args(__doc__).parse_args()
    merge(TrainingConfig(args.env_file, config_file=args.config_file, run_dir=args.run_dir))
