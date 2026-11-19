"""Training-stage adapter for the official FTS entry; no separate sampling setup."""
if __package__ in (None,''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from utils.training_config import TrainingConfig,config_args
from utils.experiment import exclusive_stage
from run_fts import sampling_arguments,run_sampling


@exclusive_stage
def sample(cfg):
    return run_sampling(cfg,sampling_arguments(cfg))


if __name__=='__main__':
    args=config_args(__doc__).parse_args()
    sample(TrainingConfig(args.env_file,config_file=args.config_file,run_dir=args.run_dir))
