"""Unified SFT / SFT+PPO training entry. CityFlow evaluation is always separate."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from utils.training_config import ROOT, TrainingConfig, atomic_json, read_json, config_args, digest
from utils.experiment import create_run, stage_state, append_jsonl


def preflight(cfg):
    if Path(sys.prefix).resolve() != ROOT/'.venv':
        raise RuntimeError('Use CoLLMLight-master/.venv/bin/python')
    if cfg['DATA_PREPARATION'] not in ('reuse','generate_teacher','sample_and_generate'):
        raise ValueError('Unknown training.data_preparation')
    if not cfg['EXPORT_AUTO'] or not cfg['EXPORT_VALIDATE']:
        raise ValueError('The total entry requires automatic validated full-model export')
    expected = 'ppo_best' if cfg['ENABLE_PPO'] else 'sft_best'
    if cfg['EXPORT_SOURCE'] not in ('auto',expected):
        raise ValueError('export.source conflicts with enable_ppo')
    if cfg['EXPORT_OUTPUT_DIR']:
        raise ValueError('The total entry exports within its unique run directory; leave output_dir null')
    if cfg['SFT_BEST_METRIC'] != 'eval_loss' or cfg['SFT_GREATER_IS_BETTER']:
        raise ValueError('SFT selection requires minimum eval_loss')
    if cfg['SFT_SAVE_STEPS']<1:raise ValueError('sft.save_steps must be positive')
    if cfg['SFT_SAVE_STRATEGY'] != cfg['SFT_EVAL_STRATEGY'] or cfg['SFT_SAVE_STRATEGY'] not in ('epoch','steps'):
        raise ValueError('SFT save/eval strategies must match (epoch or steps)')
    if not cfg.path('STUDENT_MODEL_PATH').joinpath('config.json').is_file():raise FileNotFoundError('Student model missing')
    teacher_records=None
    if not cfg['SFT_SOURCE_RUN']:
        if cfg['DATA_PREPARATION'] == 'reuse' and not cfg.path('TEACHER_OUTPUT_PATH').is_file():
            raise FileNotFoundError('Teacher data missing; explicitly select generate_teacher to make paid calls')
        if cfg['DATA_PREPARATION']=='reuse':
            teacher_records=len(read_json(cfg.path('TEACHER_OUTPUT_PATH')))
            if teacher_records<2:raise ValueError('SFT needs at least two distinct traffic records for train/validation')
        if cfg['DATA_PREPARATION'] != 'reuse':
            if cfg['TEACHER_MAX_SAMPLES']==1:raise ValueError('Total training needs at least two teacher records; use the standalone generator for one-record API probes')
            cfg.secret()
    else:
        if not cfg['ENABLE_PPO']:raise ValueError('sft_source_run is for continuing a prior SFT into PPO')
        source=cfg.path('SFT_SOURCE_RUN')
        source_manifest=read_json(source/'manifest.json')
        if source_manifest['state']!='completed' or source_manifest['training_mode'] not in ('sft_only','sft_ppo'):
            raise ValueError('Source must be a completed SFT training experiment, not a sampling/base-model probe')
        meta=read_json(source/'models/sft_best/best_model.json')
        from training.collmlight.export_best_model import file_hash
        if file_hash(source/'models/sft_best/adapter_model.safetensors') != meta['adapter_sha256']:
            raise ValueError('Source SFT weight hash mismatch')
        previous=read_json(source/'config/effective.json')
        keys=[k for k in cfg.values if k.startswith('SFT_') and k != 'SFT_SOURCE_RUN']+['STUDENT_MODEL_PATH','STUDENT_MAX_LENGTH','DATA_EVAL_RATIO']
        if any(cfg[k]!=previous[k] for k in keys):raise ValueError('Source SFT configuration mismatch')
    if cfg['ENABLE_PPO']:
        if cfg['PPO_MAX_UPDATES']!=0:raise ValueError('PPO max_updates is retired; use max_samples and num_epochs, with max_updates=0')
        importlib.metadata.version('vllm')
        from packaging.requirements import Requirement
        for line in importlib.metadata.requires('vllm') or []:
            requirement=Requirement(line)
            if requirement.name in ('transformers','torch') and not requirement.specifier.contains(importlib.metadata.version(requirement.name),prereleases=True):
                raise RuntimeError(f'vLLM dependency mismatch: {requirement}')
        if cfg['PPO_QUANTIZATION_BITS']==4:importlib.metadata.version('bitsandbytes')
        from training.collmlight.ppo_ft import validate_batch, split_records
        validate_batch(cfg)
        if cfg['DATA_PREPARATION']!='sample_and_generate':split_records(cfg)
    # One BF16 final model; PPO also retains a merged SFT base. Existing base is read-only.
    weights=sum(p.stat().st_size for p in cfg.path('STUDENT_MODEL_PATH').glob('*.safetensors'))
    copies=2 if cfg['ENABLE_PPO'] else 1
    if cfg['ENABLE_PPO'] and cfg['SFT_SOURCE_RUN']:
        source_model=cfg.path('SFT_SOURCE_RUN')/'models/trained_best_model'
        manifest=read_json(source_model/'model_manifest.json') if (source_model/'model_manifest.json').exists() else {}
        if manifest.get('selected_stage')=='sft' and source_model.stat().st_dev==ROOT.stat().st_dev:copies=1
    needed=weights*(copies+(1 if cfg['EXPORT_DTYPE']=='float32' else 0))+768*2**20
    if shutil.disk_usage(ROOT).free<needed:raise RuntimeError(f'Need at least {needed/2**30:.1f} GiB free before starting')
    return {'mode':'sft_ppo' if cfg['ENABLE_PPO'] else 'sft_only','data_preparation':cfg['DATA_PREPARATION'],
        'sft_source_run':cfg['SFT_SOURCE_RUN'] or None,'teacher_records_reused':teacher_records,'estimated_min_free_gib':needed/2**30,
        'stages':['data','sft','merge','ppo','export'] if cfg['ENABLE_PPO'] else ['data','sft','export'],
        'cityflow_evaluation':False}


def run_stage(cfg, name, module, extra=()):
    manifest=read_json(cfg.out/'manifest.json')
    if manifest['stages'].get(name,{}).get('state') in ('completed','reused'):return
    attempts=sorted((cfg.out/'logs').glob(name+'_attempt_*.log'))
    log=cfg.out/'logs'/f'{name}_attempt_{len(attempts)+1}.log'
    stage_state(cfg.out,name,'running',log=str(log.relative_to(cfg.out)))
    command=[sys.executable,'-m','training.collmlight.'+module,'--config',str(cfg.config_file),'--run-dir',str(cfg.out)]
    if cfg.env_file:command+=['--env-file',str(cfg.env_file)]
    command+=list(extra)
    env=dict(os.environ,USE_TF='0',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
    print('Stage:',name,'log:',log,flush=True)
    from utils.source_snapshot import snapshot
    before=snapshot(cfg.out/'config'/f'{name}_{len(attempts)+1}_code_before.json')
    with log.open('x') as stream:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    after=snapshot(cfg.out/'config'/f'{name}_{len(attempts)+1}_code_after.json')
    if before['files']!=after['files']:
        append_jsonl(cfg.out/'logs/events.jsonl',{'event':'code_changed_during_stage','stage':name})
    stage_state(cfg.out,name,'completed' if result.returncode==0 else 'failed',returncode=result.returncode,log=str(log.relative_to(cfg.out)))
    if result.returncode:raise RuntimeError(f'{name} failed: inspect {log}')


def train(cfg, resume=None):
    plan=preflight(cfg)
    folder=Path(resume).resolve() if resume else create_run(cfg)
    if resume:
        frozen=read_json(folder/'config/effective.json')
        expected=dict(cfg.values)
        if cfg['DATA_PREPARATION']=='sample_and_generate':expected['TRAFFIC_SAMPLE_PATH']=str(folder/'data/traffic_samples.json')
        if digest(expected)!=digest(frozen):raise ValueError('Resume configuration changed; use a new experiment')
        if read_json(folder/'manifest.json')['state']=='completed':raise ValueError('Experiment already completed')
    cfg=TrainingConfig(cfg.env_file,config_file=cfg.config_file,run_dir=folder)
    lock=folder/'.training.lock'
    import fcntl
    lock_stream=lock.open('a+')
    try:fcntl.flock(lock_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close();raise RuntimeError('This experiment already has an active training process')
    lock_stream.seek(0);lock_stream.truncate();lock_stream.write(str(os.getpid()));lock_stream.flush()
    try:
        atomic_json(folder/'reports/preflight.json',plan)
        m=read_json(folder/'manifest.json');m['state']='running';atomic_json(folder/'manifest.json',m)
        if cfg['SFT_SOURCE_RUN']:
            source=cfg.path('SFT_SOURCE_RUN')
            for rel in ['models/sft_best','data/sft_data']:
                target=folder/rel
                if not target.exists():shutil.copytree(source/rel,target)
            if not (folder/'data/selected_samples.json').exists():shutil.copy2(source/'data/selected_samples.json',folder/'data/selected_samples.json')
            m=read_json(folder/'manifest.json');m.update(parent_run=str(source),source_sft=read_json(source/'models/sft_best/best_model.json'))
            atomic_json(folder/'manifest.json',m);stage_state(folder,'sft','reused',parent_run=str(source))
        else:
            if cfg['DATA_PREPARATION']=='sample_and_generate':
                run_stage(cfg,'sample','sample_training_data')
            if cfg['DATA_PREPARATION']=='reuse':
                original=TrainingConfig(cfg.env_file,config_file=cfg.config_file).path('TEACHER_OUTPUT_PATH')
                target=cfg.path('TEACHER_OUTPUT_PATH')
                if not target.exists():shutil.copy2(original,target)
                rows=read_json(target)
                atomic_json(folder/'data/selected_samples.json',[dict(r['data'],sample_id=r['sample_id']) for r in rows])
            else:run_stage(cfg,'teacher','reasoning_tuning_data_synth')
            run_stage(cfg,'prepare','prepare_sft_data')
            run_stage(cfg,'sft','run_sft')
        if cfg['ENABLE_PPO']:
            run_stage(cfg,'merge','merge_sft_adapter')
            run_stage(cfg,'ppo','ppo_ft')
        else:stage_state(folder,'ppo','skipped_by_config')
        run_stage(cfg,'export','export_best_model')
        m=read_json(folder/'manifest.json');m.update(state='completed',final_model=str(cfg.export_dir))
        atomic_json(folder/'manifest.json',m)
        print('COMPLETED:',folder,flush=True)
        return folder
    except BaseException as exc:
        m=read_json(folder/'manifest.json');m.update(state='failed',error=type(exc).__name__)
        atomic_json(folder/'manifest.json',m)
        raise
    finally:
        fcntl.flock(lock_stream,fcntl.LOCK_UN);lock_stream.close()


def main():
    parser=config_args(__doc__);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--resume-run')
    args=parser.parse_args()
    cfg=TrainingConfig(args.env_file,config_file=args.config_file)
    if args.run_dir:parser.error('Use --resume-run for a total-entry resume')
    if args.dry_run:print(json.dumps(preflight(cfg),indent=2));return
    train(cfg,args.resume_run)

if __name__=='__main__':main()
