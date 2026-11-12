"""Route existing baseline wrappers into one immutable experiment directory."""
import contextlib
from functools import wraps
import os
from pathlib import Path
import sys
from .training_config import ROOT,TrainingConfig,atomic_json,read_json
from .experiment import create_run,append_jsonl

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:
            if not stream.closed:stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:
            if not stream.closed:stream.flush()

    def close(self):self.flush()
    def __getattr__(self,name):return getattr(self.streams[0],name)


def archive_algorithm(function):
    @wraps(function)
    def wrapper(dic_agent_conf,dic_traffic_env_conf,dic_path,roadnet,trafficflow):
        label=dic_traffic_env_conf.get('MODEL',dic_traffic_env_conf['MODEL_NAME'])
        memo=Path(dic_path.get('PATH_TO_WORK_DIRECTORY','records/default')).parent.name
        name=memo if memo not in ('','records','default') else label
        cfg=TrainingConfig(overrides={'ALGORITHM':label,'RUN_NAME':name,'ENABLE_PPO':False})
        folder=create_run(cfg)
        paths=dict(dic_path,PATH_TO_MODEL=str(folder/'models/agent'),PATH_TO_WORK_DIRECTORY=str(folder/'records/simulation'),
            PATH_TO_DATA=str((ROOT/dic_path['PATH_TO_DATA']).resolve()),PATH_TO_ERROR=str(folder/'reports/errors'),EXPERIMENT_DIR=str(folder))
        atomic_json(folder/'config/algorithm.json',{'agent':dic_agent_conf,'traffic':dic_traffic_env_conf,'paths':paths,'roadnet':roadnet,'trafficflow':trafficflow})
        wandb_dir=folder/'logs/wandb';wandb_dir.mkdir()
        old={k:os.environ.get(k) for k in ['WANDB_DIR','WANDB_MODE']}
        os.environ['WANDB_DIR']=str(wandb_dir);os.environ.setdefault('WANDB_MODE','offline')
        print('Algorithm experiment:',folder,flush=True)
        try:
            with (folder/'logs/console.log').open('x') as stream,contextlib.redirect_stdout(Tee(sys.stdout,stream)),contextlib.redirect_stderr(Tee(sys.stderr,stream)):
                result=function(dic_agent_conf,dic_traffic_env_conf,paths,roadnet,trafficflow)
            manifest=read_json(folder/'manifest.json');manifest.update(state='completed',training_mode='original_baseline',config_scope='config/algorithm.json is authoritative; effective.json is output context only')
            atomic_json(folder/'manifest.json',manifest)
            return result
        except BaseException as exc:
            manifest=read_json(folder/'manifest.json');manifest.update(state='failed',error=type(exc).__name__)
            atomic_json(folder/'manifest.json',manifest);raise
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
    return wrapper


def standalone_paths(agent,environment,paths,algorithm='LLMLight'):
    """Route older LLM baseline classes that do not use pipeline_wrapper."""
    if paths.get('EXPERIMENT_DIR'):return agent,paths
    name=Path(paths.get('PATH_TO_WORK_DIRECTORY','records/default')).parent.name
    if name in ('','records','default'):name=algorithm
    cfg=TrainingConfig(overrides={'ALGORITHM':algorithm,'RUN_NAME':name,'ENABLE_PPO':False})
    folder=create_run(cfg)
    new_paths=dict(paths,EXPERIMENT_DIR=str(folder),PATH_TO_MODEL=str(folder/'models/agent'),
        PATH_TO_WORK_DIRECTORY=str(folder/'records/simulation'),PATH_TO_DATA=str((ROOT/paths['PATH_TO_DATA']).resolve()))
    new_agent=dict(agent)
    if 'LLM_OUTPUT_DIR' in new_agent:new_agent['LLM_OUTPUT_DIR']=str(folder/'models/llm')
    if 'LOG_DIR' in new_agent:new_agent['LOG_DIR']=str(folder/'records/llm')
    atomic_json(folder/'config/algorithm.json',{'agent':new_agent,'traffic':environment,'paths':new_paths})
    os.environ['WANDB_DIR']=str(folder/'logs');os.environ.setdefault('WANDB_MODE','offline')
    return new_agent,new_paths
