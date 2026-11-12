"""Local inference telemetry without changing prompts or decisions."""
import time
import threading
from pathlib import Path
from .experiment import append_jsonl

class ObservedLLM:
    def __init__(self,client,folder):
        self.client=client;self.folder=Path(folder);self.local=threading.local();self.calls=0
    def __getattr__(self,name):return getattr(self.client,name)
    def set_context(self,**context):self.local.context=context
    def batch_ask(self,prompts,system_prompt='You are a helpful AI assistant.',**kwargs):
        context=dict(getattr(self.local,"context",{}));start=time.perf_counter()
        outputs=self.client.batch_ask(prompts,system_prompt=system_prompt,**kwargs)
        elapsed=time.perf_counter()-start;self.calls+=len(prompts)
        batch_id=f'{time.time_ns()}'
        append_jsonl(self.folder/'metrics/inference.jsonl',{'event':'batch','batch_id':batch_id,'batch_size':len(prompts),
            'batch_wall_time_s':elapsed,'backend':type(self.client).__name__,
            'generation':getattr(self.client,'llm_params',{'seed':getattr(self.client,'generation_seed',None),'max_new_tokens':getattr(self.client,'max_tokens',None),'do_sample':getattr(self.client,'do_sample',None),'temperature':0.1 if getattr(self.client,'do_sample',False) else 1.0,'top_k':50,'top_p':0.95 if getattr(self.client,'do_sample',False) else 1.0}),**{k:v for k,v in context.items() if k!='observations'}})
        native=getattr(self.client,'last_generation_metadata',[])
        for i,(prompt,text) in enumerate(zip(prompts,outputs)):
            token_meta=native[i] if len(native)==len(outputs) else {
                'completion_tokens':len(self.client.tokenizer.encode(text,add_special_tokens=False)) if hasattr(self.client,'tokenizer') else None,
                'token_count_source':'retokenized_visible_text' if hasattr(self.client,'tokenizer') else 'unavailable'}
            row={'batch_id':batch_id,'index':i,'intersection':context.get('intersections',[])[i] if i<len(context.get('intersections',[])) else None,
                **{k:v for k,v in context.items() if k not in ('intersections','observations')},**token_meta}
            snapshots=context.get('observations',[])
            if i<len(snapshots):row.update(snapshots[i])
            append_jsonl(self.folder/'metrics/tokens.jsonl',row)
            append_jsonl(self.folder/'records/llm_responses.jsonl',{'batch_id':batch_id,'index':i,'prompt':prompt,'response':text,'system':system_prompt})
        return outputs
    def ask(self,prompt,system_prompt='You are a helpful AI assistant.',**kwargs):
        return self.batch_ask([prompt],system_prompt,**kwargs)[0]
