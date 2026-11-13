# Direct script execution uses the same project root as module execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
"""LoRA-SFT using the project's installed TRL, with assistant-only labels."""
from utils.training_config import TrainingConfig, atomic_json, config_args, read_json
from pathlib import Path
import math
import shutil
from transformers import TrainerCallback, TrainerState
from utils.experiment import exclusive_stage, metrics
from training.collmlight.export_best_model import file_hash, weights_fingerprint


def tokenize_example(tokenizer, example, max_length):
    messages = [{'role': 'system', 'content': example['system']},
                {'role': 'user', 'content': example['instruction']}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': example['output']}],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat template prefix mismatch')
    if len(full) > max_length:
        raise ValueError(f'SFT example exceeds context ({len(full)} > {max_length}); no silent truncation')
    return {'input_ids': full, 'attention_mask': [1] * len(full),
            'labels': [-100] * len(prefix) + full[len(prefix):]}


class AssistantCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch
        n = max(len(f['input_ids']) for f in features)
        return {key: torch.tensor([f[key] + [pad] * (n - len(f[key])) for f in features])
                for key, pad in [('input_ids', self.tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}


def checkpoint_signature(cfg, base_weights):
    """Bind native training state to the exact data, model and training plan."""
    from importlib.metadata import version
    from utils.training_config import digest
    keys = [k for k in cfg.values if k.startswith('SFT_') and k != 'SFT_SOURCE_RUN']
    keys += ['RANDOM_SEED', 'STUDENT_MODEL_PATH', 'STUDENT_DTYPE', 'STUDENT_MAX_LENGTH']
    return digest({'settings': {k: cfg[k] for k in keys}, 'base_weights': base_weights,
        'data': {name: file_hash(cfg.data_dir/'sft_data'/name)
                 for name in ['train.json', 'eval.json', 'split_manifest.json']},
        'versions': {name: version(name) for name in ['torch', 'transformers', 'trl', 'peft', 'accelerate']}})


def resume_checkpoint(root, signature, fp16=False):
    """Require complete native state; never fall back to an adapter-only restart."""
    from transformers.trainer_utils import get_last_checkpoint
    if not root.exists():
        return None
    latest = get_last_checkpoint(str(root))
    if latest is None:
        if any(root.iterdir()):
            raise ValueError('No native SFT checkpoint in this nonempty directory; adapter-only artifacts cannot resume training')
        return None
    folder = Path(latest)
    required = ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'evaluation.json', 'best_model.json']
    if fp16:
        required.append('scaler.pt')
    missing = [name for name in required if not (folder/name).is_file() or (folder/name).stat().st_size == 0]
    if missing:
        raise ValueError(f'Incomplete SFT checkpoint {folder}: {missing}; no weights-only fallback')
    state = TrainerState.load_from_json(str(folder/'trainer_state.json'))
    meta = read_json(folder/'best_model.json')
    if state.global_step != int(folder.name.split('-')[-1]) or meta['step'] != state.global_step:
        raise ValueError('SFT checkpoint step mismatch')
    if meta['checkpoint_signature'] != signature:
        raise ValueError('SFT checkpoint data/model/training configuration or library versions changed')
    if file_hash(folder/'adapter_model.safetensors') != meta['adapter_sha256']:
        raise ValueError('SFT checkpoint adapter hash mismatch')
    if not state.best_model_checkpoint or not Path(state.best_model_checkpoint).is_relative_to(root.resolve()):
        raise ValueError('SFT checkpoint has no valid best candidate in this experiment')
    best = Path(state.best_model_checkpoint)
    if not (best/'adapter_model.safetensors').is_file() or not (best/'evaluation.json').is_file():
        raise ValueError('SFT checkpoint best candidate is missing')
    return folder


def archive_best(cfg, state):
    """Archive the candidate chosen by Trainer; do not compare scores here."""
    if not state.best_model_checkpoint:
        raise RuntimeError('Trainer did not select a validated SFT checkpoint')
    source = Path(state.best_model_checkpoint)
    meta = read_json(source/'best_model.json')
    if meta['score'] != state.best_metric or read_json(source/'evaluation.json')['eval_loss'] != state.best_metric:
        raise ValueError('Trainer best metric and checkpoint metadata differ')
    if file_hash(source/'adapter_model.safetensors') != meta['adapter_sha256']:
        raise ValueError('Trainer best adapter hash mismatch')
    target = cfg.artifact('sft_adapter')
    if (target/'best_model.json').exists() and read_json(target/'best_model.json') == meta:
        if file_hash(target/'adapter_model.safetensors') == meta['adapter_sha256']:
            return meta
    temporary = target.with_name('.sft_best_new')
    old = target.with_name('.sft_best_old')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    # Keep the best adapter lightweight; native optimizer/RNG state stays in checkpoints.
    for path in source.iterdir():
        if path.is_file() and path.name != 'trainer_state.json' and path.suffix in ('.json', '.safetensors', '.txt', '.model', '.jinja', '.md'):
            shutil.copy2(path, temporary/path.name)
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        target.rename(old)
    temporary.rename(target)
    if old.exists():
        shutil.rmtree(old)
    metrics(cfg, 'selection', {'stage': 'sft', 'event': 'new_best', **meta})
    return meta


class SFTArtifacts(TrainerCallback):
    """Telemetry and adapter archiving around the single native Trainer flow."""
    def __init__(self, cfg, signature, base_weights, resume):
        self.cfg, self.signature, self.base_weights = cfg, signature, base_weights
        self.resume = str(resume) if resume else None
        self.evaluation = None

    def on_train_begin(self, args, state, control, **kwargs):
        metrics(self.cfg, 'sft_train', {'event': 'start', 'step': state.global_step,
            'resume_from_checkpoint': self.resume, 'best_metric':state.best_metric,
            'best_model_checkpoint':state.best_model_checkpoint})
        if state.best_model_checkpoint:
            archive_best(self.cfg, state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        metrics(self.cfg, 'sft_eval' if 'eval_loss' in (logs or {}) else 'sft_train',
            {'step': state.global_step, 'epoch': state.epoch, 'resume_from_checkpoint': self.resume, **(logs or {})})

    def on_step_end(self, args, state, control, **kwargs):
        # Before Trainer reloads the best model, include a non-aligned final step.
        # Evaluation, selection and saving still run through Trainer itself.
        if args.eval_strategy == 'steps' and state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not math.isfinite(metrics['eval_loss']):
            raise RuntimeError('Non-finite SFT validation loss')
        self.evaluation = (state.global_step, dict(metrics))

    def on_save(self, args, state, control, **kwargs):
        if self.evaluation is None or self.evaluation[0] != state.global_step:
            raise RuntimeError('SFT checkpoint lacks evaluation for the same step')
        folder = Path(args.output_dir)/f'checkpoint-{state.global_step}'
        meta = {'source_run': str(self.cfg.out), 'base_weights': self.base_weights,
            'checkpoint': str(folder.relative_to(self.cfg.out)), 'step': state.global_step,
            'epoch': state.epoch, 'metric': 'eval_loss', 'score': self.evaluation[1]['eval_loss'],
            'adapter_sha256': file_hash(folder/'adapter_model.safetensors'),
            'split_sha256': file_hash(self.cfg.data_dir/'sft_data/split_manifest.json'),
            'checkpoint_signature': self.signature}
        atomic_json(folder/'evaluation.json', self.evaluation[1])
        atomic_json(folder/'best_model.json', meta)
        archive_best(self.cfg, state)

    def on_train_end(self, args, state, control, **kwargs):
        archive_best(self.cfg, state)

@exclusive_stage
def run(cfg):
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from utils.training_models import (seed_all, dtype_for, tokenizer_for, load_student,
                                       lora_for, versions, trainable_snapshot)
    seed_all(cfg['RANDOM_SEED'])
    cfg.snapshot()
    versions(cfg)
    tokenizer = tokenizer_for(cfg.path('STUDENT_MODEL_PATH'))
    tokenizer.padding_side = 'right'
    datasets = {s: Dataset.from_list([tokenize_example(tokenizer, e, cfg['STUDENT_MAX_LENGTH'])
                    for e in read_json(cfg.data_dir / 'sft_data' / f'{s}.json')]) for s in ('train', 'eval')}
    base_fingerprint=weights_fingerprint(cfg.path('STUDENT_MODEL_PATH'))
    checkpoint_root=cfg.artifact('sft_checkpoints')
    signature=checkpoint_signature(cfg,base_fingerprint)
    dtype=dtype_for(cfg)
    resume=resume_checkpoint(checkpoint_root,signature,fp16=dtype == torch.float16)
    restored=TrainerState.load_from_json(str(resume/'trainer_state.json')) if resume else None
    resumed_step=restored.global_step if restored else 0
    if resume:
        print('SFT native checkpoint:',resume,'restored step:',resumed_step,flush=True)
    if restored and resumed_step >= restored.max_steps:
        # Native training finished before the outer stage could commit its report.
        # There are no remaining updates; do not run another optimizer step.
        best=archive_best(cfg,restored)
        report={'updates':resumed_step,'new_updates':0,'resumed_step':resumed_step,
            'resume_from_checkpoint':str(resume),'train_loss':None,'eval_loss':best['score'],
            'best':best,'changed_lora_tensors':0,'finalized_completed_checkpoint':True,
            'train_examples':len(datasets['train']),'eval_examples':len(datasets['eval'])}
        cfg.report('sft_report.json',report);print(report,flush=True)
        return report
    model = load_student(cfg, cfg.path('STUDENT_MODEL_PATH'), 'SFT')
    args = SFTConfig(output_dir=str(checkpoint_root),
        max_steps=cfg['SFT_MAX_STEPS'], num_train_epochs=cfg['SFT_NUM_EPOCHS'],
        per_device_train_batch_size=cfg['SFT_PER_DEVICE_BATCH_SIZE'], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['SFT_GRADIENT_ACCUMULATION_STEPS'],
        learning_rate=cfg['SFT_LEARNING_RATE'], bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16, report_to='none',
        save_strategy=cfg['SFT_SAVE_STRATEGY'], eval_strategy=cfg['SFT_EVAL_STRATEGY'],
        save_steps=cfg['SFT_SAVE_STEPS'], eval_steps=cfg['SFT_SAVE_STEPS'],
        save_total_limit=cfg['SFT_SAVE_TOTAL_LIMIT'], save_only_model=False,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
        ignore_data_skip=False,
        logging_steps=cfg['SFT_LOGGING_STEPS'], max_seq_length=cfg['STUDENT_MAX_LENGTH'],
        gradient_checkpointing=cfg['SFT_GRADIENT_CHECKPOINTING'],
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
        seed=cfg['RANDOM_SEED'])
    trainer = SFTTrainer(model=model, args=args, tokenizer=tokenizer,
        train_dataset=datasets['train'], eval_dataset=datasets['eval'],
        peft_config=lora_for(cfg, 'SFT'), data_collator=AssistantCollator(tokenizer),
        callbacks=[SFTArtifacts(cfg,signature,base_fingerprint,resume)])
    # Capture the actually restored adapter, rather than attributing its old updates to this attempt.
    class UpdateSnapshot(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            self.before=trainable_snapshot(trainer.model)
    snapshot=UpdateSnapshot();trainer.add_callback(snapshot)
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    if trainer.state.global_step < 1:raise RuntimeError('SFT executed zero updates')
    # Trainer has now reloaded the selected best adapter. Report this fact explicitly.
    changed=[n for n,p in trainer.model.named_parameters() if n in snapshot.before and not torch.equal(p.detach().cpu(),snapshot.before[n])]
    if not resume and not any('lora_' in name for name in changed):
        raise RuntimeError('No LoRA weights changed during SFT')
    best=archive_best(cfg,trainer.state)
    report = {'updates': trainer.state.global_step, 'new_updates':trainer.state.global_step-resumed_step,
        'resumed_step':resumed_step,'resume_from_checkpoint':str(resume) if resume else None,
        'train_loss': result.training_loss,
        'train_loss_scope':'Native Trainer return for this invocation; use per-step logs for full-run curves',
        'eval_loss': best['score'], 'best':best,
        'changed_lora_tensors':len(changed), 'weight_change_reference':'restored adapter to final selected best',
        'train_examples':len(datasets['train']),'eval_examples':len(datasets['eval'])}
    cfg.report('sft_report.json',report)
    print(report)


if __name__ == '__main__':
    args = config_args(__doc__).parse_args()
    run(TrainingConfig(args.env_file, config_file=args.config_file, run_dir=args.run_dir))
