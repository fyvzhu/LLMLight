"""Standalone local inference, shared by export validation and CityFlow."""
from pathlib import Path
from .training_config import read_json


def require_full_model(path):
    path = Path(path).resolve()
    if not (path/'config.json').is_file() or (path/'adapter_config.json').exists():
        raise ValueError(f'Expected a standalone full model, not an Adapter: {path}')
    if not any((path/name).is_file() for name in ['model.safetensors','model.safetensors.index.json','pytorch_model.bin','pytorch_model.bin.index.json']):
        raise ValueError(f'Full model weights missing: {path}')
    for index in ['model.safetensors.index.json','pytorch_model.bin.index.json']:
        if (path/index).exists():
            for shard in set(read_json(path/index)['weight_map'].values()):
                if not (path/shard).is_file():raise ValueError(f'Missing model shard: {shard}')
    return path


class TransformersModel:
    def __init__(self, model_path, max_tokens=1024, dtype='auto', device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        path=require_full_model(model_path)
        self.model_path=str(path)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side='left'
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
            torch_dtype='auto' if dtype=='auto' else getattr(torch,dtype),attn_implementation='sdpa').to(device).eval()
        self.max_tokens=max_tokens
        self.calls=0
        self.responses=[]

    def batch_ask(self,user_prompts,system_prompt='You are a helpful AI assistant.',disable_thinking=False):
        import torch
        self.calls += len(user_prompts)
        if not user_prompts:return []
        prompts=[self.tokenizer.apply_chat_template([{'role':'system','content':system_prompt},
                  {'role':'user','content':p}],tokenize=False,add_generation_prompt=True,
                  enable_thinking=not disable_thinking) for p in user_prompts]
        inputs=self.tokenizer(prompts,return_tensors='pt',padding=True).to(self.model.device)
        if inputs.input_ids.shape[1]+self.max_tokens>self.model.config.max_position_embeddings:
            raise ValueError('Prompt plus requested output exceeds model context')
        with torch.inference_mode():
            outputs=self.model.generate(**inputs,max_new_tokens=self.max_tokens,do_sample=False,
                temperature=1.0,top_p=1.0,top_k=50,use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
        texts=self.tokenizer.batch_decode(outputs[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        self.responses.extend(texts)
        return texts

    def ask(self,prompt,system_prompt='You are a helpful AI assistant.',**kwargs):
        return self.batch_ask([prompt],system_prompt,**kwargs)[0]


def evaluate_policy(client, records):
    from .prompts import SYSTEM_PROMPT,get_atr_user_prompt,get_ra_user_prompt
    from .training_rewards import calculate_reactive_action_reward,calculate_advance_reasoning_reward
    values=[]
    for record in records:
        atr=client.ask(get_atr_user_prompt(record),SYSTEM_PROMPT)
        ra=client.ask(get_ra_user_prompt(record,atr),SYSTEM_PROMPT)
        score,used=calculate_reactive_action_reward(ra,record['Best_Signal'])
        values.append({'sample_id':record['sample_id'],'atr':atr,'ra':ra,'ra_reward':score,
                       'atr_reward':calculate_advance_reasoning_reward(atr,score,used)})
    if not values:raise ValueError('No validation records for full model export')
    return {'ra_reward':sum(v['ra_reward'] for v in values)/len(values),
            'atr_reward':sum(v['atr_reward'] for v in values)/len(values),'examples':values}
