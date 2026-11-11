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
                    for e in read_json(cfg.out / 'sft_data' / f'{s}.json')]) for s in ('train', 'eval')}
    model = load_student(cfg, cfg.path('STUDENT_MODEL_PATH'), 'SFT')
    args = SFTConfig(output_dir=str(cfg.out / 'sft_adapter'),
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
    trainer = SFTTrainer(model=model, args=args, tokenizer=tokenizer,
        train_dataset=datasets['train'], eval_dataset=datasets['eval'],
        peft_config=lora_for(cfg, 'SFT'), data_collator=AssistantCollator(tokenizer))
    before = trainable_snapshot(trainer.model)
    result = trainer.train()
    if trainer.state.global_step < 1:
        raise RuntimeError('SFT executed zero updates')
    changed = weight_change(trainer.model, before)
    evaluation = trainer.evaluate()
    report = {'updates': trainer.state.global_step, 'train_loss': result.training_loss,
              'eval_loss': evaluation['eval_loss'], 'changed_lora_tensors': len(changed),
              'train_examples': len(datasets['train']), 'eval_examples': len(datasets['eval'])}
    trainer.save_model(str(cfg.out / 'sft_adapter'))
    tokenizer.save_pretrained(cfg.out / 'sft_adapter')
    atomic_json(cfg.out / 'sft_report.json', report)
    print(report)


if __name__ == '__main__':
    args = config_args(__doc__).parse_args()
    run(TrainingConfig(args.env_file, config_file=args.config_file))
