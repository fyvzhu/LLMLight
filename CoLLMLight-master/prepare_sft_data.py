"""Convert teacher examples into record-split, assistant-supervised SFT data."""
import random
from utils.prompts import SYSTEM_PROMPT
from utils.training_config import TrainingConfig, atomic_json, config_args, read_json
from utils.training_rewards import parse_action


def prepare(cfg):
    records = read_json(cfg.path('TEACHER_OUTPUT_PATH'))
    ids = [r['sample_id'] for r in records]
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError('Need at least two unique completed teacher records for train/eval split')
    random.Random(cfg['RANDOM_SEED']).shuffle(ids)
    n_eval = max(1, min(len(ids) - 1, round(len(ids) * cfg['DATA_EVAL_RATIO'])))
    manifest = {'train': ids[n_eval:], 'eval': ids[:n_eval]}
    reports = []
    datasets = {'train': [], 'eval': []}
    for record in records:
        sid = record['sample_id']
        split = 'train' if sid in manifest['train'] else 'eval'
        for task in ('atr', 'ra'):
            output = record[f'{task}_output']
            reason = None
            if not isinstance(output, str) or not output.strip():
                reason = 'empty output'
            if task == 'ra':
                try:
                    parsed = parse_action(output)
                    if cfg['SFT_REQUIRE_RA_MATCH'] and parsed['signal'] != record['data']['Best_Signal']:
                        reason = 'teacher RA differs from simulator Best_Signal'
                except ValueError:
                    reason = 'invalid action JSON'
            reports.append({'sample_id': sid, 'task': task, 'split': split, 'rejected': reason})
            if reason:
                continue
            datasets[split].append({'sample_id': sid, 'task': task, 'system': SYSTEM_PROMPT,
                                    'instruction': record[f'{task}_user_prompt'], 'input': '', 'output': output})
    if not datasets['train'] or not datasets['eval']:
        raise ValueError('No usable train/eval examples after validation')
    folder = cfg.out / 'sft_data'
    for split, data in datasets.items():
        atomic_json(folder / f'{split}.json', data)
    atomic_json(folder / 'split_manifest.json', manifest)
    atomic_json(folder / 'conversion_report.json', reports)
    atomic_json(folder / 'dataset_info.json', {
        f'collmlight_{s}': {'file_name': f'{s}.json',
                          'columns': {'prompt': 'instruction', 'query': 'input',
                                      'response': 'output', 'system': 'system'}} for s in datasets})
    print({s: len(d) for s, d in datasets.items()}, 'SFT examples; RA mismatches are reported, never relabeled')
    return datasets


if __name__ == '__main__':
    args = config_args(__doc__).parse_args()
    prepare(TrainingConfig(args.env_file, config_file=args.config_file))
