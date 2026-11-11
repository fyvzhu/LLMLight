"""Run the configured full model in the original CoLLMLight/CityFlow control loop."""
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
    model_path=require_full_model(model_path or cfg.cityflow_model)
    if cfg['CITYFLOW_BACKEND']!='transformers':
        raise ValueError('This configured entry currently supports the tested local transformers backend')
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
    random.seed(cfg['CITYFLOW_SEED']);np.random.seed(cfg['CITYFLOW_SEED'])
    os.chdir(ROOT)
    rows,cols=map(int,net.split('_'));count=rows*cols
    env=deepcopy(dic_traffic_env_conf)
    env.update(NUM_AGENTS=count,NUM_INTERSECTIONS=count,NUM_ROW=rows,NUM_COL=cols,
               RUN_COUNTS=steps,ROADNET_FILE=roadnet,TRAFFIC_FILE=traffic_file,
               MODEL_NAME='CoLLMLight',PROJECT_NAME='local',
               LIST_STATE_FEATURE=['cur_phase','traffic_movement_pressure_queue'],
               DIC_REWARD_INFO={'queue_length':-0.25})
    folder=cfg.out/'cityflow_test'
    paths={'PATH_TO_MODEL':str(folder/'model'),'PATH_TO_WORK_DIRECTORY':str(folder/'records'),
           'PATH_TO_DATA':str(data_path)}
    runtime=dict(llm_path=str(model_path),llm_type='transformers',memo=cfg['RUN_NAME'],
                 dataset=dataset,debug_wollm=False,max_tokens=cfg['CITYFLOW_MAX_NEW_TOKENS'],
                 communication_threshold=0,h_w_size=5,disable_psr=False,dtype='auto')
    controller=CoLLMLight({'NEW_MAX_TOKENS':runtime['max_tokens'],'MODE':'collmlight'},env,paths,net,traffic_file,runtime)
    class LocalLogger:
        def log(self,values):atomic_json(folder/'metrics.json',values)
    metrics=controller.test(LocalLogger(),0)
    decisions=[]
    for intersection,records in controller.history_data['llm_resulst'].items():
        for record in records:
            messages=record['decision_making_messages']
            raw=messages[-1]['content']
            try:parsed=parse_action(raw)
            except ValueError:parsed=None
            decisions.append({'intersection':intersection,'step':record['step'],'action':parsed,'response':raw})
    report={'model_path':str(model_path),'backend':'transformers',
            'run_counts':steps,'llm_calls':controller.LLM.calls,
            'decisions':decisions,'valid_decisions':sum(d['action'] is not None for d in decisions),
            'fallback_decisions':metrics['LLM fail num'],'metrics':metrics}
    atomic_json(folder/'integration_report.json',report)
    print('CityFlow full model:',model_path,'LLM calls:',report['llm_calls'],
          'valid actions:',report['valid_decisions'],'fallbacks:',report['fallback_decisions'],flush=True)
    return report


if __name__=='__main__':
    parser=config_args(__doc__)
    parser.add_argument('--model_path','--model-path',dest='model_path',default=None)
    parser.add_argument('--run-counts',type=int,default=None)
    parser.add_argument('--dataset',default=None)
    parser.add_argument('--traffic_file','--traffic-file',dest='traffic_file',default=None)
    args=parser.parse_args()
    cfg=TrainingConfig(args.env_file,config_file=args.config_file)
    chosen=None
    if args.model_path:
        from pathlib import Path
        chosen=Path(args.model_path)
        if not chosen.is_absolute():chosen=ROOT/chosen
    run_cityflow(cfg,chosen,args.run_counts,args.dataset,args.traffic_file)
