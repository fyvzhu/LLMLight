"""Run the configured full model in the original CoLLMLight/CityFlow control loop."""
from pathlib import Path
from utils.training_config import TrainingConfig,ROOT,atomic_json,config_args


def run_cityflow(cfg, model_path=None, run_counts=None, dataset=None, traffic_file=None):
    import os
    import random
    import numpy as np
    from copy import deepcopy
    from utils.inference_models import require_full_model
    from utils.training_rewards import parse_action
    from utils.env_defaults import dic_traffic_env_conf
    from framework.CoLLMlight import CoLLMLight
    if not cfg.run_dir and not model_path and not cfg['CITYFLOW_MODEL_PATH']:
        raise ValueError('Evaluation requires --run-dir or cityflow_test.model_path; it never starts training')
    if model_path and cfg['CITYFLOW_MODEL_PATH'] and str(Path(model_path).resolve())!=str(cfg.path('CITYFLOW_MODEL_PATH')):
        raise ValueError('Conflicting explicit model paths')
    explicit=model_path or (cfg.path('CITYFLOW_MODEL_PATH') if cfg['CITYFLOW_MODEL_PATH'] else None)
    if cfg.run_dir and explicit is not None and Path(explicit).resolve()!=cfg.export_dir.resolve():
        raise ValueError('run-dir and explicit model identify different experiments; select just the explicit model for a new evaluation')
    model_path=require_full_model(model_path or cfg.cityflow_model)
    if cfg['CITYFLOW_BACKEND'] != 'llama':
        raise ValueError('Formal CityFlow evaluation requires the published llama API backend')
    options={'synthetic_4x4':('Synthetic','4_4'),'jinan':('Jinan','3_4'),
             'hangzhou':('Hangzhou','4_4'),'newyork_28x7':('NewYork','28_7'),
             'newyork_16x3':('NewYork','16_3')}
    dataset=dataset or cfg['CITYFLOW_DATASET']
    if dataset not in options:raise ValueError('Unknown CityFlow dataset')
    template,net=options[dataset]
    traffic_file=traffic_file or cfg['CITYFLOW_TRAFFIC_FILE']
    roadnet=f'roadnet_{net}.json'
    data_path=ROOT/'data'/template/net
    for name in (roadnet,traffic_file):
        if not (data_path/name).is_file():raise FileNotFoundError(data_path/name)
    steps=cfg['CITYFLOW_RUN_COUNTS'] if run_counts is None else run_counts
    if steps<30 or steps%30:raise ValueError('CityFlow run_counts must be a positive multiple of 30')
    cfg.values.update(CITYFLOW_DATASET=dataset,CITYFLOW_TRAFFIC_FILE=traffic_file,CITYFLOW_RUN_COUNTS=steps)
    random.seed(cfg['CITYFLOW_SEED']);np.random.seed(cfg['CITYFLOW_SEED'])
    import torch
    torch.manual_seed(cfg['CITYFLOW_SEED'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['CITYFLOW_SEED'])
    os.chdir(ROOT)
    rows,cols=map(int,net.split('_'));count=rows*cols
    env=deepcopy(dic_traffic_env_conf)
    env.update(NUM_AGENTS=count,NUM_INTERSECTIONS=count,NUM_ROW=rows,NUM_COL=cols,
               RUN_COUNTS=steps,ROADNET_FILE=roadnet,TRAFFIC_FILE=traffic_file,
               MODEL_NAME='CoLLMLight',PROJECT_NAME='local',
               LIST_STATE_FEATURE=['cur_phase','traffic_movement_pressure_queue'],
               DIC_REWARD_INFO={'queue_length':-0.25})
    from utils.experiment import create_evaluation
    if cfg.run_dir:
        parent=cfg.out
    elif cfg['CITYFLOW_MODEL_PATH'] or model_path:
        from utils.experiment import create_run
        parent=create_run(cfg)
    else:raise ValueError('Choose an explicit --run-dir or full model path')
    folder=create_evaluation(parent,dataset)
    paths={'PATH_TO_MODEL':str(folder/'model'),'PATH_TO_WORK_DIRECTORY':str(folder/'records/simulation'),
           'PATH_TO_DATA':str(data_path)}
    runtime=dict(llm_path=str(model_path),llm_type=cfg['CITYFLOW_BACKEND'],memo=cfg['RUN_NAME'],
                 dataset=dataset,debug_wollm=False,max_tokens=cfg['CITYFLOW_MAX_NEW_TOKENS'],
                 communication_threshold=0,h_w_size=5,disable_psr=cfg['CITYFLOW_DISABLE_SR'],dtype='auto',
                 port=cfg['CITYFLOW_PORT'],evaluation_dir=str(folder),seed=cfg['CITYFLOW_SEED'])
    if cfg['CITYFLOW_BACKEND']=='llama':
        import requests
        session=requests.Session();session.trust_env=False
        response=session.get(f'http://127.0.0.1:{cfg["CITYFLOW_PORT"]}/v1/models',timeout=15)
        response.raise_for_status();served=response.json()
        if not any(item.get('id')==str(model_path) or item.get('root')==str(model_path) for item in served.get('data',[])):
            raise ValueError('Inference server does not serve the requested complete model')
        atomic_json(folder/'config/server_identity.json',served)
    from utils.experiment import evaluation_logging
    with evaluation_logging(folder):
        controller=CoLLMLight({'NEW_MAX_TOKENS':runtime['max_tokens'],'MODE':'collmlight'},env,paths,net,traffic_file,runtime)
        class LocalLogger:
            def log(self,values):atomic_json(folder/'metrics.json',values)
        if cfg['CITYFLOW_SYNCHRONOUS_SR']:
            for agent in controller.agent_intersection_list:agent.syn_psr=True
        metrics=controller.test(LocalLogger(),0)
        decisions=[]
        for intersection,records in controller.history_data['llm_resulst'].items():
            for record in records:
                messages=record['decision_making_messages']
                raw=messages[-1]['content']
                try:parsed=parse_action(raw)
                except ValueError:parsed=None
                decisions.append({'intersection':intersection,'step':record['step'],'action':parsed,'response':raw})
        report={'dataset':dataset,'traffic_file':traffic_file,'roadnet_file':roadnet,'seed':cfg['CITYFLOW_SEED'],'disable_sr':cfg['CITYFLOW_DISABLE_SR'],'synchronous_sr':cfg['CITYFLOW_SYNCHRONOUS_SR'],'model_path':str(model_path),'backend':cfg['CITYFLOW_BACKEND'],
                'run_counts':steps,'llm_calls':getattr(controller.LLM,'calls',None),
                'decisions':decisions,'valid_decisions':sum(d['action'] is not None for d in decisions),
                'fallback_decisions':metrics['LLM fail num'],'metrics':metrics}
        atomic_json(folder/'reports/integration_report.json',report)
        atomic_json(folder/'config/effective.json',cfg.values)
        from training.collmlight.export_best_model import weights_fingerprint,file_hash
        atomic_json(folder/'config/data_identity.json',{'traffic_sha256':file_hash(data_path/traffic_file),'roadnet_sha256':file_hash(data_path/roadnet)})
        atomic_json(folder/'config/model_identity.json',{'model_path':str(model_path),'weights':weights_fingerprint(model_path)})
        print('CityFlow full model:',model_path,'LLM calls:',report['llm_calls'],
              'valid actions:',report['valid_decisions'],'fallbacks:',report['fallback_decisions'],flush=True)
        return report



if __name__=='__main__':
    parser=config_args(__doc__)
    parser.add_argument('--model_path','--model-path',dest='model_path',default=None)
    parser.add_argument('--run-counts',type=int,default=None)
    parser.add_argument('--backend','--model_type',dest='backend',choices=['llama'])
    parser.add_argument('--port',type=int)
    parser.add_argument('--new_max_tokens','--max-new-tokens',dest='max_tokens',type=int)
    parser.add_argument('--disable_psr',action='store_true')
    parser.add_argument('--num_rounds',type=int,default=1)
    parser.add_argument('--dataset',default=None)
    parser.add_argument('--traffic_file','--traffic-file',dest='traffic_file',default=None)
    args=parser.parse_args()
    cfg=TrainingConfig(args.env_file,config_file=args.config_file,run_dir=args.run_dir,evaluation=True)
    for name,key in [('backend','CITYFLOW_BACKEND'),('port','CITYFLOW_PORT'),('max_tokens','CITYFLOW_MAX_NEW_TOKENS')]:
        if getattr(args,name) is not None:cfg.values[key]=getattr(args,name)
    if args.disable_psr:
        cfg.values['CITYFLOW_DISABLE_SR']=True
    chosen=None
    if args.model_path:
        from pathlib import Path
        chosen=Path(args.model_path)
        if not chosen.is_absolute():chosen=ROOT/chosen
    if args.num_rounds<1:parser.error('num_rounds must be positive')
    for _ in range(args.num_rounds):run_cityflow(cfg,chosen,args.run_counts,args.dataset,args.traffic_file)
