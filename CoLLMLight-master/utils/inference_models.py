"""Standalone local inference, shared by export validation and CityFlow."""
from pathlib import Path
import threading
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
    def __init__(self, model_path, max_tokens=1024, dtype='auto', device='cuda',do_sample=False):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        path=require_full_model(model_path)
        self._metadata=threading.local()
        self.model_path=str(path)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.tokenizer.padding_side='left'
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
            torch_dtype='auto' if dtype=='auto' else getattr(torch,dtype),attn_implementation='sdpa').to(device).eval()
        self.max_tokens=max_tokens
        self.do_sample=do_sample
        self.calls=0
        self.responses=[]

    @property
    def last_generation_metadata(self):return getattr(self._metadata,'rows',[])

    @last_generation_metadata.setter
    def last_generation_metadata(self,value):self._metadata.rows=value

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
            outputs=self.model.generate(**inputs,max_new_tokens=self.max_tokens,do_sample=self.do_sample,
                temperature=0.1 if self.do_sample else 1.0,top_p=0.95 if self.do_sample else 1.0,top_k=50,use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
        texts=self.tokenizer.batch_decode(outputs[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        self.last_generation_metadata=[]
        for ids,mask in zip(outputs[:,inputs.input_ids.shape[1]:].tolist(),inputs.attention_mask):
            if self.tokenizer.eos_token_id in ids:ids=ids[:ids.index(self.tokenizer.eos_token_id)+1]
            self.last_generation_metadata.append({'prompt_tokens':int(mask.sum()),'completion_tokens':len(ids),
                'visible_completion_tokens':len(self.tokenizer.encode(self.tokenizer.decode(ids,skip_special_tokens=True),add_special_tokens=False)),
                'finish_reason':'eos' if ids and ids[-1]==self.tokenizer.eos_token_id else 'length','token_count_source':'native_generated_ids'})
        self.responses.extend(texts)
        return texts

    def ask(self,prompt,system_prompt='You are a helpful AI assistant.',**kwargs):
        return self.batch_ask([prompt],system_prompt,**kwargs)[0]
