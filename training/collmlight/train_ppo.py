"""Continue LoRA-PPO inside an existing completed SFT training directory."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.collmlight.pipeline import main

if __name__ == '__main__':
    main('ppo_only')
