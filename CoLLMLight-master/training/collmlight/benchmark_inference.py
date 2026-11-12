"""Independent SR/RD batch latency benchmark on frozen real traffic prompts."""
import argparse,time
from pathlib import Path
from utils.training_config import TrainingConfig,config_args,read_json,atomic_json
from utils.experiment import create_evaluation,append_jsonl
from utils.prompts import SYSTEM_PROMPT,get_atr_user_prompt,get_ra_user_prompt


def benchmark(cfg,repeat=2,batches=(1,5,10),warmups=1):
    from utils.inference_models import TransformersModel
    from utils.inference_logging import ObservedLLM
    from training.collmlight.ppo_ft import split_records
    folder=create_evaluation(cfg.out,'inference_benchmark')
    client=ObservedLLM(TransformersModel(cfg.cityflow_model,cfg['CITYFLOW_MAX_NEW_TOKENS'],do_sample=True),folder)
    records=split_records(cfg)[0]
    prompts={'SR':[get_atr_user_prompt(r) for r in records],
        'RD':[get_ra_user_prompt(r,'') for r in records]}
    atomic_json(folder/'config/benchmark.json',{'model':str(cfg.cityflow_model),'backend':'transformers','batches':batches,
        'warmups':warmups,'repeats':repeat,'prompt_policy':'fixed cyclic real samples; padded final batches are real prompt repetitions',
        'max_new_tokens':cfg['CITYFLOW_MAX_NEW_TOKENS'],'parameters':cfg.values,'RD_SR_input':'empty fixed SR for controlled RD latency; not a traffic effectiveness evaluation',
        'purpose':'smoke_only_truncated' if cfg['CITYFLOW_MAX_NEW_TOKENS']<=64 else 'latency_benchmark',
        'gpu_name':__import__('torch').cuda.get_device_name(0),'torch_version':__import__('torch').__version__,
        'gpu_processes_at_start':__import__('subprocess').run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],capture_output=True,text=True).stdout})
    import torch
    for module,values in prompts.items():
        for batch in batches:
            selected=[values[i%len(values)] for i in range(batch)]
            for index in range(warmups+repeat):
                client.set_context(module=module,batch_size=batch,repetition=index,warmup=index<warmups)
                torch.cuda.synchronize();start=time.perf_counter()
                client.batch_ask(selected,SYSTEM_PROMPT,disable_thinking=True)
                torch.cuda.synchronize()
                append_jsonl(folder/'metrics/latency.jsonl',{'module':module,'batch_size':batch,'repetition':index,
                    'warmup':index<warmups,'batch_wall_time_s':time.perf_counter()-start})
    print('Benchmark:',folder);return folder

if __name__=='__main__':
    p=config_args(__doc__);p.add_argument('--repeats',type=int,default=2);p.add_argument('--warmups',type=int,default=1);p.add_argument('--max-new-tokens',type=int,default=None)
    args=p.parse_args()
    if not args.run_dir:p.error('Choose --run-dir')
    cfg=TrainingConfig(args.env_file,config_file=args.config_file,run_dir=args.run_dir)
    if args.max_new_tokens is not None:cfg.values['CITYFLOW_MAX_NEW_TOKENS']=args.max_new_tokens
    benchmark(cfg,args.repeats,warmups=args.warmups)
