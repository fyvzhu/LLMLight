"""Transactional PPO metric boundaries and LoRA-only Accelerate state."""
from pathlib import Path
import os
import shutil
from utils.training_config import atomic_json, read_json, digest
from utils.experiment import append_jsonl


def durable_json(path, value):
    path=Path(path)
    atomic_json(path,value)
    with path.open('rb') as stream:os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


class MetricJournal:
    """Rollback only PPO-owned metrics; preserve SFT rows in selection.jsonl."""
    def __init__(self,cfg):self.cfg=cfg;self.root=cfg.out/'metrics'

    def files(self):
        return [p for p in self.root.glob('*.jsonl') if p.name.startswith('ppo_') or p.name=='selection.jsonl']

    def snapshot(self):
        offsets={}
        for path in self.files():
            with path.open('rb') as stream:os.fsync(stream.fileno())
            offsets[path.name]=path.stat().st_size
        return offsets

    def rollback(self,offsets):
        for name,size in offsets.items():
            if Path(name).name!=name or size<0:raise ValueError('Invalid checkpoint metric boundary')
            path=self.root/name
            if not path.exists() or path.stat().st_size<size:raise ValueError(f'Committed metrics missing or shortened: {path}')
        changed={}
        for path in self.files():
            size=offsets.get(path.name,0)
            if path.stat().st_size!=size:
                changed[path.name]={'from':path.stat().st_size,'to':size}
                with path.open('r+b') as stream:stream.truncate(size);stream.flush();os.fsync(stream.fileno())
            if size==0 and path.name not in offsets:path.unlink()
        if changed:append_jsonl(self.cfg.out/'logs/events.jsonl',{'event':'ppo_metrics_rollback','files':changed})


class TrainerState:
    """PPO state not registered by TRL with Accelerate itself."""
    def __init__(self,trainer):self.trainer=trainer

    def state_dict(self):
        t=self.trainer
        return {'kl':dict(vars(t.kl_ctl)),
                'running':{k:v for k,v in vars(t.running).items() if k!='accelerator'},
                'counters':{k:getattr(t,k) for k in ('current_step','compare_step','highest_reward') if hasattr(t,k)}}

    def load_state_dict(self,state):
        vars(self.trainer.kl_ctl).update(state['kl'])
        vars(self.trainer.running).update(state['running'])
        for k,v in state['counters'].items():setattr(self.trainer,k,v)


class PPOCheckpoint:
    """One schema: adapter/head + native optimizer/scheduler/RNG + progress."""
    def __init__(self,cfg,signature):
        self.cfg=cfg;self.signature=signature;self.root=cfg.out/'checkpoints/ppo_resume'
        self.journal=MetricJournal(cfg)

    def inspect(self,path):
        from training.collmlight.export_best_model import file_hash
        path=Path(path);marker=read_json(path/'complete.json')
        if file_hash(path/'state.json')!=marker['state_sha256']:raise ValueError('PPO checkpoint progress hash mismatch')
        state=read_json(path/'state.json')
        if state['signature']!=self.signature:raise ValueError('PPO checkpoint model/data/configuration changed')
        for name,expected in state['files'].items():
            if Path(name).name!=name or file_hash(path/name)!=expected:raise ValueError(f'PPO checkpoint file changed: {name}')
        return state

    def latest(self,epoch):
        self._clean_discarded()
        candidates=[]
        for path in self.root.glob(f'epoch_{epoch}_update_*'):
            if (path/'complete.json').is_file():
                state=self.inspect(path);candidates.append((state['updates'],path,state))
        return max(candidates,key=lambda x:x[0])[1:] if candidates else (None,None)

    def _clean_discarded(self):
        for path in self.root.glob('.discard-*'):shutil.rmtree(path)

    def _discard(self,path):
        # Hide the committed directory atomically before deleting its contents.
        # An interrupted prune must never leave a visible, half-deleted checkpoint.
        discarded=path.with_name('.discard-'+path.name)
        path.rename(discarded)
        shutil.rmtree(discarded)

    def attach(self,trainer):
        if hasattr(trainer,'_collmlight_checkpoint_state'):return
        import torch
        from peft import get_peft_model_state_dict,set_peft_model_state_dict
        from safetensors.torch import save_file,load_file
        state=TrainerState(trainer)
        trainer.accelerator.register_for_checkpointing(state)
        trainer._collmlight_checkpoint_state=state

        def save_models(models,weights,folder):
            model=trainer.accelerator.unwrap_model(trainer.model)
            adapter=get_peft_model_state_dict(model.pretrained_model)
            if not adapter or not all('lora_' in k for k in adapter):raise ValueError('Expected LoRA-only state')
            save_file({k:v.detach().contiguous().cpu() for k,v in adapter.items()},str(Path(folder)/'adapter_model.safetensors'))
            save_file({k:v.detach().contiguous().cpu() for k,v in model.v_head.state_dict().items()},str(Path(folder)/'value_head.safetensors'))
            model.pretrained_model.peft_config['default'].save_pretrained(folder)
            weights.clear()  # Never serialize the frozen base through save_state.

        def load_models(models,folder):
            model=trainer.accelerator.unwrap_model(trainer.model)
            adapter=load_file(str(Path(folder)/'adapter_model.safetensors'))
            expected=get_peft_model_state_dict(model.pretrained_model)
            if set(adapter)!=set(expected) or any(adapter[k].shape!=expected[k].shape for k in adapter):
                raise ValueError('PPO resume adapter architecture mismatch')
            set_peft_model_state_dict(model.pretrained_model,adapter)
            model.v_head.load_state_dict(load_file(str(Path(folder)/'value_head.safetensors')),strict=True)
            models.clear()

        trainer.accelerator.register_save_state_pre_hook(save_models)
        trainer.accelerator.register_load_state_pre_hook(load_models)

    def save(self,trainer,epoch,updates,next_record,rollout_id,extra=None):
        from training.collmlight.export_best_model import file_hash
        self.attach(trainer);self.root.mkdir(parents=True,exist_ok=True);self._clean_discarded()
        target=self.root/f'epoch_{epoch}_update_{updates}'
        temporary=self.root/f'.epoch_{epoch}_update_{updates}.tmp'
        if temporary.exists():shutil.rmtree(temporary)
        if target.exists():raise FileExistsError(f'Checkpoint already committed: {target}')
        trainer.accelerator.save_state(str(temporary))
        for path in temporary.iterdir():
            if path.is_file():
                with path.open('rb') as stream:os.fsync(stream.fileno())
        files={p.name:file_hash(p) for p in temporary.iterdir() if p.is_file()}
        required={'adapter_model.safetensors','value_head.safetensors','optimizer.bin','random_states_0.pkl','custom_checkpoint_0.pkl'}
        if not required.issubset(files):raise RuntimeError(f'Incomplete native PPO state: {required-set(files)}')
        state={'signature':self.signature,'epoch':epoch,'updates':updates,'next_record':next_record,
               'rollout_id':rollout_id,'metric_offsets':self.journal.snapshot(),'files':files,**(extra or {})}
        durable_json(temporary/'state.json',state)
        durable_json(temporary/'complete.json',{'state_sha256':file_hash(temporary/'state.json')})
        temporary.rename(target)
        durable_json(self.root/'latest.json',{'checkpoint':target.name,'epoch':epoch,'updates':updates})
        paths=[]
        for path in self.root.glob('epoch_*_update_*'):
            if (path/'complete.json').exists():paths.append((read_json(path/'state.json')['updates'],path))
        for _,path in sorted(paths,key=lambda x:x[0])[:-self.cfg['PPO_SAVE_TOTAL_LIMIT']]:self._discard(path)
        print(f'[PPO resume checkpoint] epoch={epoch} update={updates} next_record={next_record}: {target}',flush=True)
        return target

    def load(self,trainer,path):
        import random
        import numpy as np
        import torch
        state=self.inspect(path);self.attach(trainer)
        trainer.accelerator.load_state(str(path),load_kwargs={'weights_only':False})
        # Accelerate catches RNG loading errors internally; do not silently
        # claim a full resume if one of those states was not actually restored.
        rng=torch.load(Path(path)/'random_states_0.pkl',map_location='cpu',weights_only=False)
        numpy_state=np.random.get_state();saved_numpy=rng['numpy_random_seed']
        checks=[random.getstate()==rng['random_state'],np.array_equal(numpy_state[1],saved_numpy[1]),
                numpy_state[0]==saved_numpy[0],numpy_state[2:]==saved_numpy[2:],
                torch.equal(torch.get_rng_state(),rng['torch_manual_seed']),trainer.accelerator.step==rng['step']]
        if torch.cuda.is_available():
            current_cuda=torch.cuda.get_rng_state_all();saved_cuda=rng['torch_cuda_manual_seed']
            checks.append(len(current_cuda)==len(saved_cuda))
            checks.extend(torch.equal(a,b.cpu()) for a,b in zip(current_cuda,saved_cuda))
        if not all(checks):raise RuntimeError('Accelerate did not completely restore RNG/accumulation state')
        self.journal.rollback(state['metric_offsets'])
        print(f'[PPO resume] epoch={state["epoch"]} update={state["updates"]} next_record={state["next_record"]}',flush=True)
        return state

    def finish_epoch(self,epoch):
        self._clean_discarded()
        for path in self.root.glob(f'epoch_{epoch}_update_*'):self._discard(path)
        latest=self.root/'latest.json'
        if latest.exists() and read_json(latest)['epoch']==epoch:latest.unlink()
        if self.root.exists() and not any(self.root.iterdir()):self.root.rmdir()
