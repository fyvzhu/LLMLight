"""Explicit stage ablations. Never changes the default total training entry."""
import os
import shutil
from pathlib import Path
from utils.training_config import TrainingConfig,config_args,atomic_json,read_json
from utils.experiment import create_run,stage_state
from training.collmlight.train_collmlight import train,run_stage
from training.collmlight.export_best_model import weights_fingerprint


def plan(variant):
    return {'full':['adaptive_sft','ppo','export'], 'without_pr':['adaptive_sft','export'],
        'without_ar':['base_model','ppo','export'], 'without_both':['base_model_reference']}[variant]


def copy_full(source,target):
    def clone(a,b):
        if str(a).endswith(('.safetensors','.bin')) and Path(a).stat().st_dev==target.parent.stat().st_dev:return os.link(a,b)
        return shutil.copy2(a,b)
    shutil.copytree(source,target,copy_function=clone)


def execute(cfg,variant):
    if variant in ('full','without_pr'):
        cfg.values['ENABLE_PPO']=variant=='full'
        return train(cfg)
    cfg.values['ENABLE_PPO']=variant=='without_ar'
    cfg.values['SFT_SOURCE_RUN']=''
    folder=create_run(cfg);cfg=TrainingConfig(cfg.env_file,config_file=cfg.config_file,run_dir=folder)
    m=read_json(folder/'manifest.json');m.update(training_mode='ppo_only' if variant=='without_ar' else 'base_reference',variant=variant,
        protocol='AR stage omitted; without_ar starts PR directly from the unadapted base model, not an inferred author-private SFT protocol')
    atomic_json(folder/'manifest.json',m)
    stage_state(folder,'sft','skipped_by_ablation')
    if variant=='without_both':
        target=cfg.model_dir/'base_model_reference';copy_full(cfg.path('STUDENT_MODEL_PATH'),target)
        atomic_json(target/'reference_manifest.json',{'trained':False,'source':str(cfg.path('STUDENT_MODEL_PATH')),'weights':weights_fingerprint(target)})
        m.update(state='completed',reference_model=str(target));atomic_json(folder/'manifest.json',m)
        return folder
    weights=sum(p.stat().st_size for p in cfg.path('STUDENT_MODEL_PATH').glob('*.safetensors'))
    if shutil.disk_usage(folder).free<weights+768*2**20:raise RuntimeError('Not enough free space for PPO-only export')
    copy_full(cfg.path('STUDENT_MODEL_PATH'),cfg.artifact('sft_merged'))
    try:
        run_stage(cfg,'ppo','ppo_ft');run_stage(cfg,'export','export_best_model')
        m=read_json(folder/'manifest.json');m.update(state='completed',final_model=str(cfg.export_dir))
        atomic_json(folder/'manifest.json',m);return folder
    except BaseException:
        m=read_json(folder/'manifest.json');m['state']='failed';atomic_json(folder/'manifest.json',m);raise

if __name__=='__main__':
    p=config_args(__doc__);p.add_argument('--variant',required=True,choices=['full','without_pr','without_ar','without_both']);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:print({'variant':a.variant,'stages':plan(a.variant)})
    else:print('Ablation experiment:',execute(TrainingConfig(a.env_file,config_file=a.config_file),a.variant))
