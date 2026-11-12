# Direct script execution uses the same project root as module execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
"""LoRA-SFT using the project's installed TRL, with assistant-only labels."""
from utils.training_config import TrainingConfig, atomic_json, config_args, read_json


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


from utils.experiment import exclusive_stage

@exclusive_stage
def run(cfg):
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from utils.training_models import (seed_all, dtype_for, tokenizer_for, load_student,
                                       lora_for, versions, trainable_snapshot, weight_change)
    seed_all(cfg['RANDOM_SEED'])
    cfg.snapshot()
    versions(cfg)
    tokenizer = tokenizer_for(cfg.path('STUDENT_MODEL_PATH'))
    tokenizer.padding_side = 'right'
    datasets = {s: Dataset.from_list([tokenize_example(tokenizer, e, cfg['STUDENT_MAX_LENGTH'])
                    for e in read_json(cfg.data_dir / 'sft_data' / f'{s}.json')]) for s in ('train', 'eval')}
    model = load_student(cfg, cfg.path('STUDENT_MODEL_PATH'), 'SFT')
    import math
    import shutil
    from transformers import TrainerCallback
    from utils.experiment import metrics
    from training.collmlight.export_best_model import file_hash,weights_fingerprint
    base_fingerprint=weights_fingerprint(cfg.path('STUDENT_MODEL_PATH'))
    checkpoint_root=cfg.artifact('sft_checkpoints')
    attempt=len(list(checkpoint_root.glob('attempt_*')))+1
    checkpoints=checkpoint_root/f'attempt_{attempt}'
    args = SFTConfig(output_dir=str(checkpoints),
        max_steps=cfg['SFT_MAX_STEPS'], num_train_epochs=cfg['SFT_NUM_EPOCHS'],
        per_device_train_batch_size=cfg['SFT_PER_DEVICE_BATCH_SIZE'], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['SFT_GRADIENT_ACCUMULATION_STEPS'],
        learning_rate=cfg['SFT_LEARNING_RATE'], bf16=dtype_for(cfg) == torch.bfloat16,
        fp16=dtype_for(cfg) == torch.float16, report_to='none', save_strategy='no',
        logging_steps=cfg['SFT_LOGGING_STEPS'], max_seq_length=cfg['STUDENT_MAX_LENGTH'],
        gradient_checkpointing=cfg['SFT_GRADIENT_CHECKPOINTING'],
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
        seed=cfg['RANDOM_SEED'])
    class Selection(TrainerCallback):
        def __init__(self):self.best=None;self.saved=set()
        def on_log(self,args,state,control,logs=None,**kwargs):
            metrics(cfg,'sft_eval' if 'eval_loss' in (logs or {}) else 'sft_train',
                {'step':state.global_step,'epoch':state.epoch,**(logs or {})})
        def select(self):
            step=trainer.state.global_step
            if not step or step in self.saved:return
            evaluation=trainer.evaluate()
            score=evaluation['eval_loss']
            if not math.isfinite(score):raise RuntimeError('Non-finite SFT validation loss')
            folder=checkpoints/f'step_{step}'
            folder.mkdir(parents=True,exist_ok=False)
            trainer.save_model(str(folder));tokenizer.save_pretrained(folder)
            meta={'source_run':str(cfg.out),'base_weights':base_fingerprint,'checkpoint':str(folder.relative_to(cfg.out)),'step':step,'epoch':trainer.state.epoch,
                'metric':'eval_loss','score':score,'adapter_sha256':file_hash(folder/'adapter_model.safetensors'),
                'split_sha256':__import__('hashlib').sha256((cfg.data_dir/'sft_data/split_manifest.json').read_bytes()).hexdigest()}
            atomic_json(folder/'evaluation.json',evaluation);atomic_json(folder/'best_model.json',meta)
            if self.best is None or score<self.best['score']:
                self.best=meta
                target=cfg.artifact('sft_adapter');temporary=target.with_name('.sft_best_new')
                shutil.copytree(folder,temporary)
                old=target.with_name('.sft_best_old')
                if target.exists():target.rename(old)
                temporary.rename(target)
                if old.exists():shutil.rmtree(old)
                metrics(cfg,'selection',{'stage':'sft','event':'new_best',**meta})
            self.saved.add(step)
        def on_step_end(self,args,state,control,**kwargs):
            if cfg['SFT_SAVE_STRATEGY']=='steps' and state.global_step%cfg['SFT_SAVE_STEPS']==0:self.select()
        def on_epoch_end(self,args,state,control,**kwargs):
            if cfg['SFT_SAVE_STRATEGY']=='epoch':self.select()
    selection=Selection()
    trainer = SFTTrainer(model=model, args=args, tokenizer=tokenizer,
        train_dataset=datasets['train'], eval_dataset=datasets['eval'],
        peft_config=lora_for(cfg, 'SFT'), data_collator=AssistantCollator(tokenizer),callbacks=[selection])
    before = trainable_snapshot(trainer.model)
    result = trainer.train()
    if trainer.state.global_step < 1:raise RuntimeError('SFT executed zero updates')
    changed = weight_change(trainer.model, before)
    selection.select()
    if selection.best is None:raise RuntimeError('No validated SFT candidate')
    report = {'updates': trainer.state.global_step, 'train_loss': result.training_loss,
        'eval_loss': selection.best['score'], 'best':selection.best,'changed_lora_tensors':len(changed),
        'train_examples':len(datasets['train']),'eval_examples':len(datasets['eval'])}
    cfg.report('sft_report.json',report)
    print(report)


if __name__ == '__main__':
    args = config_args(__doc__).parse_args()
    run(TrainingConfig(args.env_file, config_file=args.config_file, run_dir=args.run_dir))
