"""W&B is optional: always retain local records when it is not installed."""
import importlib.util
import os
from pathlib import Path
from .experiment import append_jsonl

if importlib.util.find_spec('wandb') is not None:
    import wandb
else:
    class LocalTracker:
        def init(self,**kwargs):
            folder=Path(os.environ.get('WANDB_DIR','logs/local_tracking'));folder.mkdir(parents=True,exist_ok=True)
            append_jsonl(folder/'events.jsonl',{'event':'init','settings':kwargs})
            return self
        def log(self,values):
            folder=Path(os.environ.get('WANDB_DIR','logs/local_tracking'))
            append_jsonl(folder/'events.jsonl',{'event':'log','values':values})
        def Table(self,columns,data):return {'columns':columns,'data':data}
        def finish(self):pass
    wandb=LocalTracker()
