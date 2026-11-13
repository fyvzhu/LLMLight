"""Official FTS entry and the single sampling configuration builder."""
import os,sys,argparse
from copy import deepcopy

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--memo", type=str, default='MALLMlight-GPT3.5')
    # parser.add_argument("--model_type", type=str, default="gpt") #/gpt/llama
    # parser.add_argument("--model_path", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--memo", type=str, default='FTs_avgql')
    # parser.add_argument("--memo", type=str, default='default strategy')
    parser.add_argument("--model_type", type=str, default="gpt") #/gpt/llama
    parser.add_argument("--model_path", type=str, default=None) 
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--new_max_tokens", type=int, default=2048)
    parser.add_argument("--proj_name", type=str, default="LLM-TSCS")
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--multi_process", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="synthetic_4x4")
    parser.add_argument("--traffic_file", type=str, default="anon_4_4_synthetic_8000.json")
    parser.add_argument("--mode", type=str, default="fts", help="fts")
    parser.add_argument("--reward_type", type=str, default="avg_ql", help="near_ql/near_wt/near_tt/avg_ql/avg_wt/avg_tt/far_ql/far_wt/far_tt/qlwt/early_ocp/far_ocp/avg_ocp")
    parser.add_argument("--long_info", type=bool, default=True)
    parser.add_argument("--feed_back", type=bool, default=True)
    parser.add_argument("--feed_back_num", type=int, default=1)
    parser.add_argument("--debug_wollm", type=bool, default=False)
    parser.add_argument("--reward_period", type=int, default=5)
    parser.add_argument("--ignore_threshold", type=float, default=0)
    parser.add_argument('--max_sampling_steps',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--config',dest='config_file',default=None)
    parser.add_argument('--run-dir',default=None)
    arguments=sys.argv[1:] if argv is None else list(argv)
    result=parser.parse_args(arguments)
    result._explicit={parser._option_string_actions[token.split('=',1)[0]].dest for token in arguments if token.split('=',1)[0] in parser._option_string_actions}
    return result

SAMPLING_CONFIG_FIELDS = {'max_sampling_steps': 'SAMPLING_MAX_STEPS', 'dataset': 'SAMPLING_DATASET', 'traffic_file': 'SAMPLING_TRAFFIC_FILE', 'reward_type': 'SAMPLING_REWARD_TYPE', 'reward_period': 'SAMPLING_REWARD_PERIOD', 'ignore_threshold': 'SAMPLING_IGNORE_THRESHOLD', 'long_info': 'SAMPLING_LONG_INFO', 'feed_back': 'SAMPLING_FEED_BACK', 'feed_back_num': 'SAMPLING_FEED_BACK_NUM', 'debug_wollm': 'SAMPLING_DEBUG'}


def sampling_arguments(cfg,overrides=None):
    args=parse_args([])
    for argument,key in SAMPLING_CONFIG_FIELDS.items():setattr(args,argument,cfg[key])
    args.seed=cfg['RANDOM_SEED'];args.memo=cfg['RUN_NAME']
    if overrides is not None:
        for name in overrides._explicit-{'config_file','run_dir'}:setattr(args,name,getattr(overrides,name))
    return args


def build_sampling_config(in_args,cfg):
    from utils.env_defaults import dic_traffic_env_conf
    from utils.training_config import ROOT
    traffic_file_list = [] 

    if in_args.dataset == 'jinan':
        count = 3600
        # count = 86400
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json", "anon_3_4_jinan_synthetic_24000_60min.json", 
                             "anon_3_4_jinan_synthetic_24h.json", "anon_3_4_jinan_synthetic_24h_6000.json"]
        template = "Jinan"
    elif in_args.dataset == 'hangzhou':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json", "anon_4_4_hangzhou_synthetic_32000_60min.json"]
        template = "Hangzhou"
    elif in_args.dataset == 'newyork_16x3':
        count = 3600
        road_net = "16_3"
        traffic_file_list = ["anon_16_3_newyork_real.json"]
        template = "NewYork"
    elif in_args.dataset == 'newyork_28x7':
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        template = "NewYork"
    elif in_args.dataset == 'synthetic_4x4':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_synthetic.json", "anon_4_4_synthetic_5000.json", "anon_4_4_synthetic_8000.json", 'anon_4_4_synthetic_10000.json']
        template = "Synthetic"
    if "24h" in in_args.traffic_file:
        count = 86400

    in_args.model = in_args.memo

    if in_args.traffic_file not in traffic_file_list:
        raise ValueError('Traffic file is not valid for the selected dataset')
    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(in_args.traffic_file)

    dic_agent_conf_extra = {
        # "LLM_PATH": in_args.llm_path,
        # "LLM_MODEL": in_args.llm_model,
        # "LOG_DIR": f"./{in_args.llm_model}_logs",
        "NEW_MAX_TOKENS": in_args.new_max_tokens,
        "MODE": in_args.mode
    }

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,

        # "MODEL_NAME": f"{in_args.model}-{dic_agent_conf_extra['LLM_MODEL']}",
        "PROJECT_NAME": in_args.proj_name,
        "RUN_COUNTS": count,
        "NUM_ROUNDS": in_args.num_rounds,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,

        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],

        "DIC_REWARD_INFO": {
            'queue_length': -0.25
        }
    }
    my_config = {
        'llm_path': in_args.model_path,
        'llm_type': in_args.model_type,
        'memo': in_args.memo,
        'dataset': in_args.traffic_file.strip('.json'),
        'accumulate_patience': 5,
        'congestion_scale_ratio': 0.5, # hyper_param
        'boundary_distance_threshold': 4,
        'reward_type': in_args.reward_type,
        'long_info': in_args.long_info,
        'feed_back': in_args.feed_back,
        'feed_back_num': in_args.feed_back_num,
        'debug_wollm': in_args.debug_wollm,
        'reward_period':in_args.reward_period,
        'ignore_threshold': in_args.ignore_threshold,
    }

    if in_args.eightphase:
        dic_traffic_env_conf_extra["PHASE"] = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0]
        }
        dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                    'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30, 30, 30, 30, 30]

    else:
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30]

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]
    dic_path_extra = {
        'PATH_TO_MODEL':str(cfg.out/'tmp/sampling/model'),
        'PATH_TO_WORK_DIRECTORY':str(cfg.out/'tmp/sampling/work'),
        'PATH_TO_LOG':str(cfg.out/'records/sampling'),
        'PATH_TO_DATA':str(ROOT/'data'/template/road_net),
    }
    env=deepcopy(dic_traffic_env_conf);env.update(dic_traffic_env_conf_extra)
    env['MODEL_NAME']='FTS'
    if in_args.max_sampling_steps<0:raise ValueError('max_sampling_steps must be nonnegative')
    my_config.update(max_sampling_steps=in_args.max_sampling_steps or count//env['MIN_ACTION_TIME'],
        sample_output=str(cfg.path('TRAFFIC_SAMPLE_PATH')),evaluation_dir=str(cfg.out))
    return dic_agent_conf_extra,env,dic_path_extra,my_config,f'{template}-{road_net}',in_args.traffic_file.split('.')[0]


def run_sampling(cfg,args):
    import random
    import numpy as np
    from framework.FTSample import FTS
    from utils.training_config import ROOT
    from utils.experiment import stage_summary,clean_workspace
    if not cfg.run_dir:raise ValueError('Sampling requires a managed output directory')
    output=cfg.path('TRAFFIC_SAMPLE_PATH')
    if output.exists():raise FileExistsError(f'Refusing to overwrite existing samples: {output}')
    random.seed(args.seed);np.random.seed(args.seed);os.chdir(ROOT)
    agent,env,paths,options,roadnet,flow=build_sampling_config(args,cfg)
    stage_summary(cfg,'sample',{'settings':{k:getattr(args,k) for k in SAMPLING_CONFIG_FIELDS},'seed':args.seed})
    sampler=FTS(agent,env,paths,roadnet,flow,options)
    class LocalLogger:
        def log(self,values):stage_summary(cfg,'sample',{'seed':args.seed,'settings':{k:getattr(args,k) for k in SAMPLING_CONFIG_FIELDS},'metrics':values})
    result=sampler.test(LocalLogger(),0)
    print('Sampling complete:',output,'records:',len(sampler.FT_data),flush=True)
    clean_workspace(cfg.out,cfg.out/'tmp/sampling')
    return result


def main(in_args):
    from utils.training_config import TrainingConfig,read_json,atomic_json
    from utils.experiment import create_run,exclusive_stage,stage_state,training_log
    cfg=TrainingConfig(config_file=in_args.config_file,run_dir=in_args.run_dir)
    args=sampling_arguments(cfg,in_args)
    if cfg.run_dir:
        # A managed run's frozen sampling settings cannot be silently replaced.
        if any(getattr(args,k)!=cfg[v] for k,v in SAMPLING_CONFIG_FIELDS.items()) or args.seed!=cfg['RANDOM_SEED']:
            raise ValueError('Sampling overrides differ from the managed run configuration')
    else:
        for argument,key in SAMPLING_CONFIG_FIELDS.items():cfg.values[key]=getattr(args,argument)
        cfg.values.update(RANDOM_SEED=args.seed,ENABLE_PPO=False,DATA_PREPARATION='sample_and_generate')
        folder=create_run(cfg);cfg=TrainingConfig(config_file=cfg.config_file,run_dir=folder)
    print('Sampling run:',cfg.out,flush=True)
    manifest=read_json(cfg.out/'manifest.json');manifest.update(state='running',training_mode='sampling_only')
    atomic_json(cfg.out/'manifest.json',manifest)
    try:
        import contextlib,sys
        from utils.baseline_archive import Tee
        stage_state(cfg.out,'sample','running',attempt=1,log='logs/train.log')
        with training_log(cfg,'stage=sample attempt=1') as stream,contextlib.redirect_stdout(Tee(sys.stdout,stream)),contextlib.redirect_stderr(Tee(sys.stderr,stream)):
            result=exclusive_stage(run_sampling)(cfg,args)
        stage_state(cfg.out,'sample','completed')
        manifest=read_json(cfg.out/'manifest.json');manifest['state']='completed';atomic_json(cfg.out/'manifest.json',manifest)
        return result
    except BaseException:
        stage_state(cfg.out,'sample','failed')
        manifest=read_json(cfg.out/'manifest.json');manifest['state']='failed';atomic_json(cfg.out/'manifest.json',manifest)
        raise


if __name__=='__main__':main(parse_args())
