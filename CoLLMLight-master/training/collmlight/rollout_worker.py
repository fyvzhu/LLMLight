"""Isolated original vLLM ATR-then-RA epoch rollout; preserves exact token IDs."""
import argparse
import time
from pathlib import Path
from utils.training_config import atomic_json, read_json
from utils.prompts import SYSTEM_PROMPT,get_atr_user_prompt,get_ra_user_prompt
from utils.training_rewards import calculate_reactive_action_reward,calculate_advance_reasoning_reward
from utils.experiment import append_jsonl


def generate(base,adapter,groups,settings,metrics_path):
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    args=dict(model=base,max_model_len=settings['STUDENT_MAX_LENGTH'],dtype='bfloat16',tensor_parallel_size=1,
        gpu_memory_utilization=settings['VLLM_GPU_MEMORY'],enforce_eager=True,disable_log_stats=True,
        max_num_seqs=max(1,min(16,(settings.get('PPO_MAX_SAMPLES',4) or 16))),enable_lora=bool(adapter))
    if adapter:args['max_lora_rank']=settings['PPO_LORA_RANK']
    llm=LLM(**args)
    sampling=SamplingParams(temperature=settings['PPO_TEMPERATURE'],top_k=settings['PPO_TOP_K'],
        top_p=settings['PPO_TOP_P'],max_tokens=settings['PPO_MAX_NEW_TOKENS'],
        stop_token_ids=[tokenizer.eos_token_id],seed=settings['RANDOM_SEED'])
    request=LoRARequest('policy',1,adapter) if adapter else None
    result={}
    for split,records in groups.items():
        if not records:result[split]=[];continue
        rows=[dict(data=r) for r in records]
        for task in ('atr','ra'):
            prompts=[get_atr_user_prompt(r['data']) if task=='atr' else get_ra_user_prompt(r['data'],r['atr']['text']) for r in rows]
            formatted=[tokenizer.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in prompts]
            for p in formatted:
                if len(tokenizer.encode(p,add_special_tokens=False))+sampling.max_tokens>settings['STUDENT_MAX_LENGTH']:
                    raise ValueError('Prompt plus generation exceeds context; no silent truncation')
            start=time.perf_counter();outputs=llm.generate(formatted,sampling,lora_request=request);elapsed=time.perf_counter()-start
            append_jsonl(metrics_path,{'split':split,'module':'SR' if task=='atr' else 'RD','batch_size':len(rows),
                'batch_wall_time_s':elapsed,'backend':'vllm','adapter':adapter,'temperature':sampling.temperature})
            for row,out in zip(rows,outputs):
                o=out.outputs[0]
                row[task]={'text':o.text,'query_ids':out.prompt_token_ids,'response_ids':list(o.token_ids),'finish_reason':o.finish_reason}
                append_jsonl(metrics_path,{'split':split,'module':'SR' if task=='atr' else 'RD','sample_id':row['data']['sample_id'],
                    'prompt_tokens':len(out.prompt_token_ids),'completion_tokens':len(o.token_ids),'finish_reason':o.finish_reason})
        for row in rows:
            ra,used=calculate_reactive_action_reward(row['ra']['text'],row['data']['Best_Signal'])
            row.update(ra_reward=ra,atr_reward=calculate_advance_reasoning_reward(row['atr']['text'],ra,used))
        result[split]=rows
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();request=read_json(args.request)
    atomic_json(args.output,generate(**request))

if __name__=='__main__':main()
