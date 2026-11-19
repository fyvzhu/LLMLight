"""Current-epoch, small-batch generation cache with metric commit boundaries."""
import json
import os
from pathlib import Path
from tqdm import tqdm
from utils.training_config import read_json
from utils.experiment import metrics
from training.collmlight.ppo_checkpoint import MetricJournal, durable_json

PHASES=('train_atr','train_ra','eval_atr','eval_ra')


class RolloutCache:
    def __init__(self,cfg,epoch,signature):
        self.cfg=cfg;self.epoch=epoch;self.signature=signature
        self.folder=cfg.out/'tmp/ppo_rollouts'/f'epoch_{epoch}'
        self.rollout_id=f'epoch_{epoch}_{signature[:16]}'
        self.journal=MetricJournal(cfg);self.progress_path=self.folder/'progress.json'

    def recover(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        if self.progress_path.exists():
            self.state=read_json(self.progress_path)
            if self.state['signature']!=self.signature:raise ValueError('Cached rollout model/data/generation configuration changed')
            self.journal.rollback(self.state['metric_offsets'])
        else:
            self.state={'signature':self.signature,'rollout_id':self.rollout_id,'epoch':self.epoch,
                'counts':{p:0 for p in PHASES},'offsets':{p:0 for p in PHASES},
                'metric_offsets':self.journal.snapshot(),'complete':False}
            durable_json(self.progress_path,self.state)
        for phase in PHASES:
            path=self.folder/(phase+'.jsonl');size=self.state['offsets'][phase]
            if size and (not path.exists() or path.stat().st_size<size):raise ValueError('Committed generation cache missing or shortened')
            if path.exists():
                with path.open('r+b') as stream:stream.truncate(size);stream.flush();os.fsync(stream.fileno())
        return self

    def read(self,phase,records):
        path=self.folder/(phase+'.jsonl')
        rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if len(rows)!=self.state['counts'][phase] or [r['sample_id'] for r in rows]!=[r['sample_id'] for r in records[:len(rows)]]:
            raise ValueError('Generation cache record ordering changed')
        return [row['response'] for row in rows]

    def commit(self,phase,records,responses,statistics,updates):
        start=self.state['counts'][phase]
        if len(records)!=len(responses) or len(records)!=len(statistics['completion_tokens']):
            raise ValueError('Generation returned an incomplete batch')
        path=self.folder/(phase+'.jsonl')
        with path.open('ab') as stream:
            for row,response,tokens in zip(records,responses,statistics['completion_tokens']):
                stream.write((json.dumps({'sample_id':row['sample_id'],'response':response,'completion_tokens':tokens},ensure_ascii=False)+'\n').encode())
            stream.flush();os.fsync(stream.fileno())
        metrics(self.cfg,'ppo_inference',{'epoch':self.epoch,'global_update':updates,'rollout_id':self.rollout_id,
            'phase':phase,'module':'SR' if phase.endswith('atr') else 'RD','split':phase.split('_')[0],
            'purpose':'training_rollout','batch_start':start,'batch_size':len(records),**statistics})
        self.state['counts'][phase]+=len(records);self.state['offsets'][phase]=path.stat().st_size
        self.state['metric_offsets']=self.journal.snapshot()
        durable_json(self.progress_path,self.state)

    def finish(self,output):
        train_count=len(output['train']);eval_count=len(output.get('evaluation',{}).get('examples',[]))
        if any(self.state['counts'][p]!=(train_count if p.startswith('train') else eval_count) for p in PHASES):
            raise ValueError('Cannot commit an incomplete ATR/RA rollout')
        durable_json(self.folder/'output.json',output)
        self.state['complete']=True;self.state['metric_offsets']=self.journal.snapshot()
        from training.collmlight.export_best_model import file_hash
        self.state['output_sha256']=file_hash(self.folder/'output.json')
        durable_json(self.progress_path,self.state)

    def output(self):
        from training.collmlight.export_best_model import file_hash
        state=read_json(self.progress_path)
        if state['signature']!=self.signature or not state['complete']:raise ValueError('No complete matching rollout cache')
        if file_hash(self.folder/'output.json')!=state['output_sha256']:raise ValueError('Completed rollout cache was modified')
        return read_json(self.folder/'output.json')


def generate_cached(generator,cache,phase,records,atr,updates):
    from utils.prompts import get_atr_user_prompt_wsys,get_ra_user_prompt_wsys
    responses=cache.read(phase,records)
    print(f'[PPO generation] epoch={cache.epoch} {phase}: cached={len(responses)}/{len(records)}',flush=True)
    progress=tqdm(total=len(records),initial=len(responses),desc=f'PPO {cache.epoch} {phase}',unit='record',dynamic_ncols=True)
    try:
        for start in range(len(responses),len(records),cache.cfg['PPO_ROLLOUT_BATCH_SIZE']):
            batch=records[start:start+cache.cfg['PPO_ROLLOUT_BATCH_SIZE']]
            prompts=([get_atr_user_prompt_wsys(row) for row in batch] if phase.endswith('atr') else
                     [get_ra_user_prompt_wsys(row,atr[start+i]) for i,row in enumerate(batch)])
            outputs=generator.generate_batch(prompts)
            cache.commit(phase,batch,outputs,generator.last_statistics,updates)
            responses.extend(outputs);progress.update(len(batch))
    finally:progress.close()
    return responses


def generate_epoch(cfg,epoch,signature,train,evaluation,adapter,updates):
    from training.collmlight.ppo_ft import VLLMLocalGenerator,assemble_rollout,score_evaluation
    cache=RolloutCache(cfg,epoch,signature).recover()
    if cache.state['complete']:
        print(f'[PPO generation] complete cache reused: {cache.folder}',flush=True)
        return cache.output()
    generator=VLLMLocalGenerator(str(cfg.artifact('sft_merged')),adapter,cfg['STUDENT_MAX_LENGTH'],cfg)
    try:
        atr=generate_cached(generator,cache,'train_atr',train,None,updates)
        ra=generate_cached(generator,cache,'train_ra',train,atr,updates)
        eval_atr=generate_cached(generator,cache,'eval_atr',evaluation,None,updates)
        eval_ra=generate_cached(generator,cache,'eval_ra',evaluation,eval_atr,updates)
        result={'train':assemble_rollout(train,atr,ra),'evaluation':score_evaluation(evaluation,eval_atr,eval_ra)}
        cache.finish(result)
        return result
    finally:generator.unload_model()
