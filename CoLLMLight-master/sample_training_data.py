"""Small real CityFlow/FTS sampling run; no teacher or LLM dependencies."""
import os
import random
from copy import deepcopy
from pathlib import Path
from utils.training_config import ROOT, TrainingConfig, atomic_json, config_args


def sample(cfg, steps=4):
    import numpy as np
    from framework.FTSample import FTS
    from utils.env_defaults import dic_traffic_env_conf
    random.seed(cfg['RANDOM_SEED'])
    np.random.seed(cfg['RANDOM_SEED'])
    os.chdir(ROOT)
    output = cfg.path('TRAFFIC_SAMPLE_PATH')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing samples: {output}')
    env_conf = deepcopy(dic_traffic_env_conf)
    env_conf.update(NUM_AGENTS=16, NUM_INTERSECTIONS=16, NUM_ROW=4, NUM_COL=4,
                    RUN_COUNTS=3600, TRAFFIC_FILE='anon_4_4_synthetic_8000.json',
                    ROADNET_FILE='roadnet_4_4.json', MODEL_NAME='FTS', PROJECT_NAME='local',
                    LIST_STATE_FEATURE=['cur_phase', 'traffic_movement_pressure_queue'],
                    DIC_REWARD_INFO={'queue_length': -0.25})
    paths = {'PATH_TO_MODEL': str(cfg.out / 'sampling_model'),
             'PATH_TO_WORK_DIRECTORY': str(cfg.out / 'sampling_records'),
             'PATH_TO_DATA': 'data/Synthetic/4_4'}
    options = dict(dataset='synthetic_4x4', memo=cfg['RUN_NAME'], reward_type='avg_ql',
                   boundary_distance_threshold=4, ignore_threshold=0, long_info=True,
                   feed_back=True, feed_back_num=1, debug_wollm=False, reward_period=5,
                   accumulate_patience=5, congestion_scale_ratio=0.5,
                   max_sampling_steps=steps, sample_output=str(output))
    sampler = FTS({'NEW_MAX_TOKENS': 2048, 'MODE': 'fts'}, env_conf, paths,
                  'Synthetic-4_4', env_conf['TRAFFIC_FILE'], options)
    class LocalLogger:
        def log(self, values):
            atomic_json(cfg.out / 'sampling_metrics.json', values)
    result = sampler.test(LocalLogger(), 0)
    print('Sampling complete:', output, 'records:', len(sampler.FT_data))
    return result


if __name__ == '__main__':
    parser = config_args(__doc__)
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    sample(TrainingConfig(args.env_file, config_file=args.config_file), args.steps)
