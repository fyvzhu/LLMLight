"""On-policy ATR/RA LoRA-PPO using TRL 0.9.2 in the existing project environment."""
import math
from pathlib import Path
from utils.prompts import SYSTEM_PROMPT, get_atr_user_prompt, get_ra_user_prompt
from utils.training_config import TrainingConfig, atomic_json, config_args, read_json
from utils.training_rewards import calculate_reactive_action_reward, calculate_advance_reasoning_reward


def validate_batch(cfg):
    batch = cfg['PPO_BATCH_SIZE']
    divisor = cfg['PPO_MINI_BATCH_SIZE'] * cfg['PPO_GRADIENT_ACCUMULATION_STEPS']
    if batch % 2 or batch % divisor:
        raise ValueError('PPO_BATCH_SIZE counts ATR/RA queries, must be even and divisible by mini_batch_size * gradient_accumulation_steps')
    return batch // 2


def save_checkpoint(trainer, tokenizer, folder, state):
    import random
    import numpy as np
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(str(folder))
    tokenizer.save_pretrained(folder)
    torch.save({'optimizer': trainer.optimizer.state_dict(),
                'scheduler': trainer.lr_scheduler.state_dict() if trainer.lr_scheduler else None,
                'value_head': trainer.accelerator.unwrap_model(trainer.model).v_head.state_dict(),
                'kl_value': trainer.kl_ctl.value, 'current_step': trainer.current_step,
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'state': state}, folder / 'training_state.pt')
    atomic_json(folder / 'progress.json', state)


def restore_training(trainer, folder):
    import random
    import numpy as np
    import torch
    state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=False)
    trainer.optimizer.load_state_dict(state['optimizer'])
    if trainer.lr_scheduler and state['scheduler']:
        trainer.lr_scheduler.load_state_dict(state['scheduler'])
    trainer.accelerator.unwrap_model(trainer.model).v_head.load_state_dict(state['value_head'])
    trainer.kl_ctl.value = state['kl_value']
    trainer.current_step = state['current_step']
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['state']


def build_trainer(cfg, resume=None):
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    from utils.training_models import load_student, tokenizer_for, lora_for
    tokenizer = tokenizer_for(cfg.out / 'sft_merged')
    tokenizer.padding_side = 'left'
    base = load_student(cfg, cfg.out / 'sft_merged', 'PPO')
    if resume:
        from peft import PeftModel
        policy = PeftModel.from_pretrained(base, resume, is_trainable=True)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(policy)
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(base, peft_config=lora_for(cfg, 'PPO'))
    if cfg['PPO_GRADIENT_CHECKPOINTING']:
        model.pretrained_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.pretrained_model.enable_input_require_grads()
    # Dropout must not change between rollout, old-policy evaluation and reference.
    import torch
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.0
    config = PPOConfig(batch_size=cfg['PPO_BATCH_SIZE'], mini_batch_size=cfg['PPO_MINI_BATCH_SIZE'],
        gradient_accumulation_steps=cfg['PPO_GRADIENT_ACCUMULATION_STEPS'],
        learning_rate=cfg['PPO_LEARNING_RATE'], ppo_epochs=cfg['PPO_OPTIMIZATION_EPOCHS'],
        seed=cfg['RANDOM_SEED'], log_with=None, whiten_rewards=False,
        optimize_device_cache=True)
    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer, ref_model=None)
    if not trainer.is_peft_model:
        raise RuntimeError('Expected PEFT policy: reference must disable only the new PPO adapter')
    return trainer, tokenizer


def rollout(trainer, tokenizer, cfg, record, greedy=False):
    import torch
    from utils.training_models import query_ids
    trainer.model.eval()
    queries, responses, texts = [], [], []
    for task in ('atr', 'ra'):
        prompt = get_atr_user_prompt(record) if task == 'atr' else get_ra_user_prompt(record, texts[0])
        ids = query_ids(tokenizer, SYSTEM_PROMPT, prompt, cfg['STUDENT_MAX_LENGTH'], cfg['PPO_MAX_NEW_TOKENS'])
        query = torch.tensor(ids, device=trainer.accelerator.device, dtype=torch.long)
        kwargs = dict(max_new_tokens=cfg['PPO_MAX_NEW_TOKENS'], do_sample=not greedy,
                      eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                      use_cache=True)
        if not greedy: kwargs.update(temperature=cfg['PPO_TEMPERATURE'], top_p=0.95, top_k=50)
        with torch.no_grad():
            response = trainer.generate(query, return_prompt=False, **kwargs).squeeze(0).detach()
        if response.numel() == 0: raise RuntimeError('Empty generated response')
        queries.append(query)
        responses.append(response)
        texts.append(tokenizer.decode(response, skip_special_tokens=True))
    ra, used = calculate_reactive_action_reward(texts[1], record['Best_Signal'])
    atr = calculate_advance_reasoning_reward(texts[0], ra, used)
    return queries, responses, [atr, ra], texts


def evaluate(trainer, tokenizer, cfg, records):
    if not records: raise ValueError('PPO eval set is empty')
    totals = [0.0, 0.0]
    examples = []
    for record in records:
        _, _, rewards, texts = rollout(trainer, tokenizer, cfg, record, greedy=True)
        totals = [a+b for a, b in zip(totals, rewards)]
        examples.append({'sample_id': record['sample_id'], 'atr': texts[0], 'ra': texts[1],
                         'atr_reward': rewards[0], 'ra_reward': rewards[1]})
    return {'atr_reward': totals[0]/len(records), 'ra_reward': totals[1]/len(records), 'examples': examples}


def run(cfg, resume=None, additional_updates=None):
    import torch
    from utils.training_models import seed_all, versions, trainable_snapshot, weight_change
    per_batch = validate_batch(cfg)
    manifest = read_json(cfg.out / 'sft_data/split_manifest.json')
    selected = read_json(cfg.out / 'selected_samples.json')
    train = [r for r in selected if r['sample_id'] in manifest['train']]
    evaluation = [r for r in selected if r['sample_id'] in manifest['eval']]
    if len(train) < per_batch or not evaluation:
        raise ValueError('Insufficient PPO records for one complete train batch and validation')
    seed_all(cfg['RANDOM_SEED'])
    versions(cfg)
    cfg.snapshot()
    trainer, tokenizer = build_trainer(cfg, resume)
    state = {'updates': 0, 'cursor': 0, 'best_ra': -1e30}
    if resume:
        state = restore_training(trainer, Path(resume))
    else:
        atomic_json(cfg.out / 'ppo_initial_evaluation.json', evaluate(trainer, tokenizer, cfg, evaluation))
    before = trainable_snapshot(trainer.model)
    frozen = {n: p.detach().view(-1)[:64].cpu().clone() for n, p in trainer.model.named_parameters() if not p.requires_grad}
    start = state['updates']
    if additional_updates is not None:
        if additional_updates <= 0: raise ValueError('additional_updates must be positive')
        limit = start + additional_updates
    else:
        scheduled = (len(train)//per_batch) * cfg['PPO_NUM_EPOCHS']
        limit = min(scheduled, cfg['PPO_MAX_UPDATES']) if cfg['PPO_MAX_UPDATES'] > 0 else scheduled
    if limit <= start: raise ValueError('No PPO updates scheduled')
    history = []
    while state['updates'] < limit:
        queries, responses, rewards, examples = [], [], [], []
        for _ in range(per_batch):
            record = train[state['cursor'] % len(train)]
            state['cursor'] += 1
            q, r, scores, texts = rollout(trainer, tokenizer, cfg, record)
            queries.extend(q); responses.extend(r)
            rewards.extend(torch.tensor(v, device=trainer.accelerator.device) for v in scores)
            examples.append({'sample_id': record['sample_id'], 'atr': texts[0], 'ra': texts[1], 'rewards': scores})
        if len(queries) != cfg['PPO_BATCH_SIZE']: raise RuntimeError('PPO batch mismatch')
        stats = trainer.step(queries, responses, rewards)
        for key, value in stats.items():
            if key.startswith('ppo/loss/') and not torch.isfinite(torch.as_tensor(value)).all():
                raise RuntimeError(f'Non-finite PPO loss: {key}')
        state['updates'] += 1
        # Evaluate this exact updated policy before selecting/saving it.
        result = evaluate(trainer, tokenizer, cfg, evaluation)
        improved = result['ra_reward'] > state['best_ra']
        if improved: state['best_ra'] = result['ra_reward']
        folder = cfg.out / 'ppo_checkpoints' / f"update_{state['updates']}"
        save_checkpoint(trainer, tokenizer, folder, state)
        if improved:
            save_checkpoint(trainer, tokenizer, cfg.out / 'ppo_best', state)
            atomic_json(cfg.out / 'ppo_best/evaluation.json', result)
            atomic_json(cfg.out / 'ppo_best/best_model.json', {
                'checkpoint': f"ppo_checkpoints/update_{state['updates']}",
                'update': state['updates'], 'metric': 'ra_reward', 'score': result['ra_reward'],
                'base_model': 'sft_merged',
                'validation_sample_ids': [r['sample_id'] for r in evaluation]})
        report = {'update': state['updates'], 'query_count': len(queries),
                  'policy_loss': float(stats['ppo/loss/policy']),
                  'value_loss': float(stats['ppo/loss/value']),
                  'evaluation': result, 'rollouts': examples}
        history.append(report)
        atomic_json(cfg.out / f"ppo_update_{state['updates']}.json", report)
        print('PPO update', state['updates'], 'policy loss', report['policy_loss'],
              'eval RA reward', result['ra_reward'], flush=True)
    changed = weight_change(trainer.model, before)
    for name, p in trainer.model.named_parameters():
        if name in frozen and not torch.equal(p.detach().view(-1)[:64].cpu(), frozen[name]):
            raise RuntimeError('Frozen SFT reference base weights changed')
    atomic_json(cfg.out / ('ppo_resume_report.json' if resume else 'ppo_report.json'), {
        'resumed_from': str(Path(resume).resolve()) if resume else None,
        'updates_this_run': state['updates'] - start, 'total_updates': state['updates'],
        'changed_trainable_tensors': len(changed), 'frozen_base_unchanged': True,
        'checkpoint': str(folder), 'history': history})
    if cfg['EXPORT_AUTO']:
        # Release GPU training objects before loading export/evaluation models.
        import gc
        del trainer, tokenizer, before, frozen
        gc.collect()
        torch.cuda.empty_cache()
        from export_best_model import export_best
        export_best(cfg)


if __name__ == '__main__':
    parser = config_args(__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--additional-updates', type=int)
    args = parser.parse_args()
    run(TrainingConfig(args.env_file, config_file=args.config_file), args.resume, args.additional_updates)
