"""Export the validation-selected PPO adapter as an independently loadable full model."""
import gc
import hashlib
import shutil
import tempfile
from pathlib import Path
from utils.training_config import TrainingConfig, atomic_json, config_args, read_json, digest


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def weights_fingerprint(path):
    files=sorted(p for p in Path(path).iterdir() if p.suffix in ('.safetensors','.bin') and not p.name.startswith('training_'))
    return {p.name:file_hash(p) for p in files}


def action_map(evaluation):
    from utils.training_rewards import parse_action
    actions={}
    for row in evaluation['examples']:
        try:actions[row['sample_id']]=parse_action(row['ra'])['signal']
        except ValueError:actions[row['sample_id']]=None
    return actions


def best_source(cfg):
    if cfg['EXPORT_SOURCE']!='ppo_best':raise ValueError('export.source must be ppo_best; latest is not necessarily best')
    source=cfg.out/'ppo_best'
    evaluation=read_json(source/'evaluation.json')
    progress=read_json(source/'progress.json')
    meta=read_json(source/'best_model.json') if (source/'best_model.json').exists() else {
        'checkpoint':f"ppo_checkpoints/update_{progress['updates']}",
        'update':progress['updates'],'score':progress['best_ra'],'metric':'ra_reward','base_model':'sft_merged'}
    checkpoint=(cfg.out/meta['checkpoint']).resolve()
    if not checkpoint.is_relative_to((cfg.out/'ppo_checkpoints').resolve()):raise ValueError('Best checkpoint must belong to this experiment')
    if evaluation['ra_reward']!=meta['score'] or progress['best_ra']!=meta['score']:
        raise ValueError('Best checkpoint score/evaluation mismatch')
    if file_hash(source/'adapter_model.safetensors')!=file_hash(checkpoint/'adapter_model.safetensors'):
        raise ValueError('Best adapter differs from its recorded checkpoint')
    return source,meta,evaluation


def export_best(cfg):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from utils.inference_models import TransformersModel,evaluate_policy,require_full_model
    from utils.training_models import tokenizer_for
    if not cfg['EXPORT_VALIDATE']:raise ValueError('A trained_best_model export requires export.validate=true')
    source,meta,evaluation=best_source(cfg)
    base=require_full_model(cfg.out/'sft_merged')
    target=cfg.export_dir
    if target.resolve() in (source.resolve(),base.resolve()) or target.resolve().is_relative_to(source.resolve()):
        raise ValueError('Export destination must be separate from source weights')
    actual_dtype=read_json(base/'config.json').get('torch_dtype','float32') if cfg['EXPORT_DTYPE']=='auto' else cfg['EXPORT_DTYPE']
    if actual_dtype not in ('float32','float16','bfloat16'):raise ValueError('Invalid export dtype')
    provenance={'base_weights':weights_fingerprint(base), 'adapter_sha256':file_hash(source/'adapter_model.safetensors'),
                'checkpoint':meta['checkpoint'],'evaluation':evaluation,'dtype':actual_dtype,
                'max_new_tokens':cfg['PPO_MAX_NEW_TOKENS']}
    signature=digest(provenance)
    if target.exists():
        manifest=read_json(target/'model_manifest.json')
        if manifest['source_signature']!=signature:raise FileExistsError('Export source changed; select a new export.output_dir')
        require_full_model(target)
        if weights_fingerprint(target)!=manifest['exported_weights']:raise ValueError('Export weights were modified')
        previous=read_json(target/'export_validation_report.json')
        if action_map(previous['source_validation'])!=action_map(previous['exported_validation']):
            raise ValueError('Exported validation decisions differ from source checkpoint')
        print('Verified existing standalone export:',target)
        return target
    if shutil.disk_usage(target.parent if target.parent.exists() else cfg.out).free < sum(p.stat().st_size for p in base.glob('*.safetensors'))*2:
        raise RuntimeError('Insufficient disk space for safe temporary model export')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.export-',dir=target.parent))
    try:
        tokenizer=tokenizer_for(base)
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32,
            attn_implementation='sdpa').to(cfg['EXPORT_DEVICE'])
        model=PeftModel.from_pretrained(model,source).eval()
        inputs=tokenizer('Return JSON with a traffic signal ETWT.',return_tensors='pt').to(cfg['EXPORT_DEVICE'])
        with torch.inference_mode():before=model(**inputs).logits[:,-1].float().cpu()
        merged=model.merge_and_unload(safe_merge=True).eval()
        with torch.inference_mode():after=merged(**inputs).logits[:,-1].float().cpu()
        max_error=(before-after).abs().max().item()
        if not torch.allclose(before,after,atol=5e-4,rtol=5e-4):raise RuntimeError(f'FP32 merge mismatch {max_error}')
        merged.to(getattr(torch,actual_dtype)).save_pretrained(temporary,safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        del merged,model,inputs,before,after
        gc.collect();torch.cuda.empty_cache()
        client=TransformersModel(temporary,cfg['PPO_MAX_NEW_TOKENS'],dtype=actual_dtype)
        with torch.inference_mode():
            saved_logits=client.model(**client.tokenizer('Return JSON with a traffic signal ETWT.',return_tensors='pt').to(client.model.device)).logits[:,-1].float().cpu()
        # Compare saved precision against the same precision base+adapter.
        reference=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=getattr(torch,actual_dtype),
            attn_implementation='sdpa').to('cuda')
        reference=PeftModel.from_pretrained(reference,source).eval()
        with torch.inference_mode():
            reference_logits=reference(**client.tokenizer('Return JSON with a traffic signal ETWT.',return_tensors='pt').to('cuda')).logits[:,-1].float().cpu()
        saved_error=(saved_logits-reference_logits).abs().max().item()
        tolerance=5e-4 if actual_dtype=='float32' else 0.3
        if not torch.allclose(saved_logits,reference_logits,atol=tolerance,rtol=0.02 if actual_dtype!='float32' else 5e-4):
            raise RuntimeError(f'Saved precision logits mismatch {saved_error}')
        del reference
        gc.collect();torch.cuda.empty_cache()
        selected=read_json(cfg.out/'selected_samples.json')
        ids=read_json(cfg.out/'sft_data/split_manifest.json')['eval']
        records=[r for r in selected if r['sample_id'] in ids]
        revalidated=evaluate_policy(client,records)
        if revalidated['ra_reward']!=evaluation['ra_reward'] or action_map(revalidated)!=action_map(evaluation):
            atomic_json(cfg.out/'export_failed_validation.json',{'source':evaluation,'exported':revalidated,'dtype':actual_dtype})
            raise RuntimeError('Saved precision changed validation decisions; inspect export_failed_validation.json and choose a suitable export.dtype')
        report={'fp32_merge_max_error':max_error,'saved_precision_max_error':saved_error,
                'saved_dtype':actual_dtype,'source_validation':evaluation,'exported_validation':revalidated,
                'validation_score_matches':True,'independent_full_model_load':True}
        atomic_json(temporary/'export_validation_report.json',report)
        manifest={'source_signature':signature,'best_checkpoint':meta['checkpoint'],'best_update':meta['update'],
                  'validation_metric':meta['metric'],'validation_score':meta['score'],'dtype':actual_dtype,
                  'source_base':str(base),'source_adapter':str(source),
                  'exported_weights':weights_fingerprint(temporary),'standalone':True,'validated':True}
        atomic_json(temporary/'model_manifest.json',manifest)
        del client
        gc.collect();torch.cuda.empty_cache()
        temporary.rename(target)
        atomic_json(cfg.out/'export_report.json',{'model_path':str(target),'manifest':manifest,'validation':report})
        print('Exported validated standalone best model:',target,flush=True)
        return target
    finally:
        if temporary.exists():shutil.rmtree(temporary)


if __name__=='__main__':
    args=config_args(__doc__).parse_args()
    export_best(TrainingConfig(args.env_file,config_file=args.config_file))
