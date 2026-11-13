# Direct script execution uses the same project root as module execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
"""Export the validation-selected PPO adapter as an independently loadable full model."""
import gc
import os
import sys
import subprocess
import hashlib
import shutil
import tempfile
from pathlib import Path
from utils.training_config import ROOT, TrainingConfig, atomic_json, config_args, read_json, digest


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def weights_fingerprint(path):
    files=sorted(p for p in Path(path).iterdir() if p.suffix in ('.safetensors','.bin') and not p.name.startswith('training_'))
    return {p.name:file_hash(p) for p in files}








def best_source(cfg):
    if not cfg.run_dir:raise ValueError('Choose a training run with --run-dir')
    stage='ppo' if cfg['ENABLE_PPO'] else 'sft'
    source=cfg.artifact(stage+'_best')
    meta=read_json(source/'best_model.json')
    checkpoint=(cfg.out/meta['checkpoint']).resolve()
    if stage=='ppo' and not checkpoint.is_relative_to(cfg.artifact('ppo_checkpoints').resolve()):
        raise ValueError('Best checkpoint escapes this experiment')
    # Reused SFT best has its original checkpoint recorded; source weights are copied and hash-checked.
    if stage=='sft' and cfg['SFT_SOURCE_RUN']:
        checkpoint=cfg.path('SFT_SOURCE_RUN')/meta['checkpoint']
    if not checkpoint.is_file() and not checkpoint.is_dir():raise FileNotFoundError(checkpoint)
    if file_hash(source/'adapter_model.safetensors')!=meta['adapter_sha256']:
        raise ValueError('Selected best adapter hash mismatch')
    if file_hash(checkpoint/'adapter_model.safetensors')!=meta['adapter_sha256']:
        raise ValueError('Best source differs from its selected checkpoint')
    evaluation=read_json(source/'evaluation.json')
    key='eval_loss' if stage=='sft' else 'ra_reward'
    if meta['metric']!=key or evaluation[key]!=meta['score']:
        raise ValueError('Best validation metric/score mismatch')
    return source,meta,evaluation


from utils.experiment import exclusive_stage

@exclusive_stage
def export_best(cfg):
    if not cfg.run_dir:raise ValueError('Choose a training run with --run-dir')
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from utils.inference_models import TransformersModel,require_full_model
    from utils.training_models import tokenizer_for
    source,meta,selection=best_source(cfg)
    stage='ppo' if cfg['ENABLE_PPO'] else 'sft'
    base=cfg.artifact('sft_merged') if stage=='ppo' else cfg.path('STUDENT_MODEL_PATH')
    target=cfg.export_dir
    signature=digest({'base':weights_fingerprint(base),'adapter':meta['adapter_sha256'],'stage':stage,'dtype':cfg['EXPORT_DTYPE'],'integrity_protocol':'official_ppo_eval_v1'})
    if target.exists():
        manifest=read_json(target/'model_manifest.json')
        if manifest['source_signature']!=signature or weights_fingerprint(target)!=manifest['exported_weights']:
            raise ValueError('Existing export does not match this best checkpoint')
        require_full_model(target);return target
    target.parent.mkdir(parents=True,exist_ok=True)
    source_bytes=sum(p.stat().st_size for p in base.glob('*.safetensors'))
    scale=2 if cfg['EXPORT_DTYPE']=='float32' else 1
    if shutil.disk_usage(target.parent).free<source_bytes*scale+128*2**20:
        raise RuntimeError('Insufficient free space for the selected full-model export dtype')
    temp=Path(tempfile.mkdtemp(prefix='.export-',dir=target.parent))
    try:
        tokenizer=tokenizer_for(base)
        model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to(cfg['EXPORT_DEVICE'])
        model=PeftModel.from_pretrained(model,source).eval()
        inputs=tokenizer('Return a traffic signal: ETWT',return_tensors='pt').to(cfg['EXPORT_DEVICE'])
        with torch.inference_mode():before=model(**inputs).logits[:,-1].float().cpu()
        model=model.merge_and_unload(safe_merge=True).eval()
        with torch.inference_mode():after=model(**inputs).logits[:,-1].float().cpu()
        error=(before-after).abs().max().item()
        if not torch.allclose(before,after,atol=5e-4,rtol=5e-4):raise RuntimeError(f'FP32 merge mismatch: {error}')
        dtype='bfloat16' if cfg['EXPORT_DTYPE']=='auto' else cfg['EXPORT_DTYPE']
        if dtype not in ('float32','bfloat16','float16'):raise ValueError('Invalid export dtype')
        model.to(getattr(torch,dtype)).save_pretrained(temp,safe_serialization=True)
        tokenizer.save_pretrained(temp)
        del model,inputs,before,after;gc.collect();torch.cuda.empty_cache()
        client=TransformersModel(temp,cfg['STUDENT_MAX_NEW_TOKENS'],dtype=dtype)
        tokenizer_matches=client.tokenizer.get_vocab()==tokenizer.get_vocab() and client.tokenizer.chat_template==tokenizer.chat_template and client.tokenizer.eos_token_id==tokenizer.eos_token_id
        if not tokenizer_matches:raise RuntimeError('Exported tokenizer differs from the selected base')
        expected_config=read_json(base/'config.json');actual_config=client.model.config.to_dict()
        keys=('model_type','vocab_size','hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads')
        config_matches=all(actual_config.get(key)==expected_config.get(key) for key in keys)
        if not config_matches:raise RuntimeError('Exported architecture configuration differs from base')
        with torch.inference_mode():saved_logits=client.model(**client.tokenizer('Return a traffic signal: ETWT',return_tensors='pt').to(client.model.device)).logits[:,-1].float().cpu()
        reference=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=getattr(torch,dtype),attn_implementation='sdpa').to('cuda')
        reference=PeftModel.from_pretrained(reference,source).eval()
        with torch.inference_mode():ref_logits=reference(**client.tokenizer('Return a traffic signal: ETWT',return_tensors='pt').to('cuda')).logits[:,-1].float().cpu()
        saved_error=(ref_logits-saved_logits).abs().max().item()
        if not torch.allclose(ref_logits,saved_logits,atol=0.3 if dtype!='float32' else 5e-4,rtol=0.02 if dtype!='float32' else 5e-4):
            raise RuntimeError(f'Saved precision merge mismatch: {saved_error}')
        del reference;gc.collect();torch.cuda.empty_cache()
        # Release local integrity-check models before starting the vLLM process.
        del client,saved_logits,ref_logits;gc.collect();torch.cuda.empty_cache()
        output=temp/'official_inference_check.json'
        attempts=len(list((cfg.out/'logs').glob('export_inference_attempt_*.log')))+1
        logfile=cfg.out/'logs'/f'export_inference_attempt_{attempts}.log'
        env=dict(os.environ,USE_TF='0',VLLM_USE_V1='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,'-m','training.collmlight.ppo_ft','--config',str(cfg.config_file),'--run-dir',str(cfg.out),
            '--evaluate-model',str(temp),'--output',str(output)]
        if cfg.env_file:command+=['--env-file',str(cfg.env_file)]
        with logfile.open('x') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Official-protocol export integrity inference failed: {logfile}')
        validated=read_json(output)
        report={'independent_full_model_load':True,'fp32_merge_max_error':error,'saved_precision_max_error':saved_error,
            'tokenizer_matches_base':tokenizer_matches,'model_config_matches_base':config_matches,
            'saved_dtype':dtype,'selection_validation':selection,'official_inference_check':validated,
            'protocol_note':'Integrity only. Uses published PPO inference; never reselects best or supplies CityFlow ATT/AWT/AQL.'}
        manifest={'source_signature':signature,'training_mode':read_json(cfg.out/'manifest.json')['training_mode'],
            'selected_stage':stage,'best_checkpoint':meta['checkpoint'],'validation_metric':meta['metric'],
            'validation_score':meta['score'],'source_base':str(base),'source_adapter':str(source),
            'source_adapter_sha256':meta['adapter_sha256'],'dtype':dtype,'exported_weights':weights_fingerprint(temp),'standalone':True,'validated':True}
        atomic_json(temp/'model_manifest.json',manifest);atomic_json(temp/'export_validation_report.json',report)
        require_full_model(temp);temp.rename(target)
        cfg.report('export_report.json',{'model_path':str(target),'manifest':manifest,'validation':report})
        print('Exported best full model:',target,flush=True)
        return target
    finally:
        if temp.exists():shutil.rmtree(temp)


if __name__=='__main__':
    args=config_args(__doc__).parse_args()
    export_best(TrainingConfig(args.env_file,config_file=args.config_file,run_dir=args.run_dir))
