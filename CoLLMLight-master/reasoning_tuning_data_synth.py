"""Generate resumable teacher ATR/RA examples from real FTS records."""
import random
from utils.LLMs import GPT_model
from utils.prompts import SYSTEM_PROMPT, get_atr_user_prompt, get_ra_user_prompt
from utils.prompts import get_advance_traffic_reasoning_prompt_adaptive
from utils.prompts import get_historical_observation_atr, get_current_observation_atr
from utils.training_config import TrainingConfig, atomic_json, config_args, digest, read_json
from utils.training_rewards import SIGNALS, parse_action


def eligible(record):
    return (record.get('Timestep', 0) > 0 and bool(record.get('Traffic_state_history'))
            and bool(record.get('Current_Observation', {}).get('non_empty_lanes'))
            and record.get('Best_Signal') in SIGNALS
            and set(SIGNALS).issubset(record.get('Signal_Rank', {})))


def select_samples(records, maximum, seed):
    unique = {digest(r): r for r in records if eligible(r)}
    keys = sorted(unique)
    random.Random(seed).shuffle(keys)
    return [dict(unique[k], sample_id=k) for k in keys[:maximum]]


def synthesize(cfg, client=None):
    selected = select_samples(read_json(cfg.path('TRAFFIC_SAMPLE_PATH')),
                              cfg['TEACHER_MAX_SAMPLES'], cfg['TEACHER_RANDOM_SEED'])
    if not selected:
        raise ValueError('No eligible FTS samples; run sample_training_data.py first')
    signature = digest({'selected': selected, 'teacher': {
        k: v for k, v in cfg.values.items() if k.startswith('TEACHER_') and k != 'TEACHER_RESUME'}})
    progress_path = cfg.out / 'teacher_progress.json'
    progress = {'signature': signature, 'items': {}}
    if cfg['TEACHER_RESUME'] and progress_path.exists():
        progress = read_json(progress_path)
        if progress['signature'] != signature:
            raise ValueError('Teacher inputs/config changed; use a new RUN_NAME or disable TEACHER_RESUME')
    atomic_json(cfg.out / 'selected_samples.json', selected)
    already_complete = all('ra_output' in progress['items'].get(r['sample_id'], {}) for r in selected)
    cfg.snapshot()
    client = client or GPT_model(
        model=cfg['TEACHER_MODEL'], key=cfg.secret(),
        url=cfg['TEACHER_BASE_URL'].rstrip('/') + '/chat/completions',
        timeout=cfg['TEACHER_TIMEOUT_SECONDS'], max_attempts=cfg['TEACHER_MAX_ATTEMPTS'],
        use_env_proxy=cfg['TEACHER_USE_ENV_PROXY'],
        llm_params={'temperature': cfg['TEACHER_TEMPERATURE'],
                    'max_tokens': cfg['TEACHER_MAX_TOKENS'],
                    'enable_thinking': cfg['TEACHER_ENABLE_THINKING']})
    for data in selected:
        sample_id = data['sample_id']
        item = progress['items'].get(sample_id, {})
        if 'ra_output' in item:
            continue
        level = 'easy'
        if len(data['Traffic_state_history']) > 1:
            for lane, state in data['Traffic_state_history'][-2].items():
                if ('upstream' in lane or 'downstream' in lane) and state['Occupancy (%)'] >= 40:
                    level = 'complex'
        if 'atr_output' not in item:
            t1 = data['Timestep']
            adaptive = get_advance_traffic_reasoning_prompt_adaptive(
                t1 - 1, t1, get_historical_observation_atr(t1 - 1, data['Traffic_state_history']),
                get_current_observation_atr(t1, data['Traffic_state_history']), level)
            response = client.ask(adaptive, SYSTEM_PROMPT)
            if not response.strip():
                raise ValueError('Empty ATR')
            item = {'sample_id': sample_id, 'atr_user_prompt': get_atr_user_prompt(data),
                    'atr_output': response, 'data': data, 'complex_level': level,
                    'teacher_atr_prompt': adaptive}
            progress['items'][sample_id] = item
            atomic_json(progress_path, progress)
        prompt = get_ra_user_prompt(data, item['atr_output'])
        response = client.ask(prompt, SYSTEM_PROMPT)
        parse_action(response)
        item.update(ra_user_prompt=prompt, ra_output=response)
        atomic_json(progress_path, progress)
        completed = [progress['items'][r['sample_id']] for r in selected
                     if 'ra_output' in progress['items'].get(r['sample_id'], {})]
        atomic_json(cfg.path('TEACHER_OUTPUT_PATH'), completed)
        print(f'Teacher completed {len(completed)}/{len(selected)}', flush=True)
    result = [progress['items'][r['sample_id']] for r in selected]
    atomic_json(cfg.path('TEACHER_OUTPUT_PATH'), result)
    report = {'selected': len(selected), 'completed': len(result),
              'requests_this_run': client.requests_count, 'usage_this_run': client.usage}
    atomic_json(cfg.out / ('teacher_resume_report.json' if already_complete else 'teacher_report.json'), report)
    print(report)
    return result


if __name__ == '__main__':
    args = config_args(__doc__).parse_args()
    synthesize(TrainingConfig(args.env_file, config_file=args.config_file))
