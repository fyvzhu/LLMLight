"""Published ICLR 2026 teacher generation, configured for this project."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import re
import random
import time
from utils.LLMs import GPT_model
from utils.prompts import *
from utils.training_config import TrainingConfig,atomic_json,read_json,config_args
from utils.experiment import exclusive_stage,append_jsonl
LLM = None

def extract_json(response):
    try:
        match = re.search(r"({.*})", response.strip(), re.DOTALL)
        if match:
            json_data = match.group(1)  # Extract the JSON body
            data = json.loads(json_data)
            return data
        else:
            print("No JSON data found in the response.")
            return ""

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Parse error: {e}")

def convert_dict_to_numbered_text(input_dict):
    numbered_lines = [
        f"{index + 1}. **{key}**:\n{value}\n"
        for index, (key, value) in enumerate(input_dict.items())
    ]

    # Join the text lines into a single paragraph
    return "\n".join(numbered_lines)

def gpt4_json_reponse_generate(prompt, system_prompt):
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = LLM.ask(prompt, system_prompt)
            # Try to parse the framework
            data = extract_json(response)
            # If parsing is successful, return the data
            return data

        except Exception as e:
            # Catch parsing failure exceptions
            retries += 1
            print(f"Parse error: {e}, retrying... ({retries}/{max_retries})")
    raise RuntimeError("Maximum retries reached. Failed to parse framework.")

def get_historical_observation_atr(timestep, Traffic_state_history):
    # get the history before timestep t-1
    h_w_size = 5

    history_state = Traffic_state_history[-h_w_size-1:-1]

    timestep_start = timestep - len(history_state)
    text = "## Historical Observation\n"
    text += "- Lanes include both signal-controlled lanes at the current intersection (e.g., NL, NT, SL, ST, EL, ET, WL, WT) and upstream/downstream lanes from neighboring intersections (e.g., SL's upstream lane (4, ST)). **Only lanes with vehicles are shown.**\n"
    text += "- Values are shown as: value or value(+change from previous timestep).\n\n"
    for i in range(len(history_state)):
        text += f"Timestep {timestep_start + i} signal: {history_state[i]['Signal']}\n"
        text += f"Timestep {timestep_start + i + 1} traffic states:\n"
        text += "|Lane|Cars Input|Cars Output|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy (%)|\n"
        signal_consequence = history_state[i]
        for lane in signal_consequence:
            if lane in ['Signal', 'Summary']:
                continue
            lane_data = signal_consequence[lane]
            exist = lane_data['Occupancy (%)'] + lane_data['Cars Input'] + lane_data['Cars Output']
            if not exist:
                continue
            text += f"|{lane}|{int(lane_data['Cars Input'])}|{int(lane_data['Cars Output'])}|{int(lane_data['Queued Cars'])}({'+' if lane_data['Queued Cars Change']>=0 else ''}{int(lane_data['Queued Cars Change'])})|{int(lane_data['Moving Cars'])}({'+' if lane_data['Moving Cars Change']>=0 else ''}{int(lane_data['Moving Cars Change'])})|{lane_data['Average Waiting Time (mins)']}({'+' if lane_data['Average Waiting Time Change (mins)']>=0 else ''}{lane_data['Average Waiting Time Change (mins)']})|{lane_data['Occupancy (%)']}({'+' if lane_data['Occupancy Change (%)']>=0 else ''}{lane_data['Occupancy Change (%)']})|\n"
        text += '\n'
    if len(history_state) > 0:
        return text
    else:
        return ""

def get_current_observation_atr(step_num, Traffic_state_history):
    signal = Traffic_state_history[-1]['Signal']
    text = "## Current Observation\n"
    text += f"Timestep {step_num} signal: {signal}\n"
    return text

def get_current_observation_ra(step_num, Traffic_state_history):
    signal = Traffic_state_history[-1]['Signal']
    text = "# Current Observation\n"
    text += "- Lanes include both signal-controlled lanes at the current intersection (e.g., NL, NT, SL, ST, EL, ET, WL, WT) and upstream/downstream lanes from neighboring intersections (e.g., SL's upstream lane (4, ST)). **Only lanes with vehicles are shown.**\n"
    text += "- Values are shown as: value or value(+change from previous timestep).\n\n"
    text += f"Timestep {step_num-1} signal: {signal}\n"
    current_state = Traffic_state_history[-1]
    text += f"Timestep {step_num} traffic states:\n"
    text += "|Lane|Cars Input|Cars Output|Queued Cars|Moving Cars|Average Waiting Time (mins)|Occupancy (%)|\n"
    for lane in current_state:
        if lane in ['Signal', 'Summary']:
            continue
        lane_data = current_state[lane]
        exist = lane_data['Occupancy (%)'] + lane_data['Cars Input'] + lane_data['Cars Output']
        if not exist:
            continue
        text += f"|{lane}|{int(lane_data['Cars Input'])}|{int(lane_data['Cars Output'])}|{int(lane_data['Queued Cars'])}({'+' if lane_data['Queued Cars Change']>=0 else ''}{int(lane_data['Queued Cars Change'])})|{int(lane_data['Moving Cars'])}({'+' if lane_data['Moving Cars Change']>=0 else ''}{int(lane_data['Moving Cars Change'])})|{lane_data['Average Waiting Time (mins)']}({'+' if lane_data['Average Waiting Time Change (mins)']>=0 else ''}{lane_data['Average Waiting Time Change (mins)']})|{lane_data['Occupancy (%)']}({'+' if lane_data['Occupancy Change (%)']>=0 else ''}{lane_data['Occupancy Change (%)']})|\n"
    text += '\n'
    return text

def get_local_signal_priority_text(signal_value_dict):
    text = "# Local Signal Priority\n"
    text += "This rank only considers your own intersection and assumes the downstream allows the release.\n"
    text += "|Rank|Signal|Waiting Time Reduction|\n"
    for i, p in enumerate(sorted(signal_value_dict, key=signal_value_dict.get, reverse=True)):
        text += "|{}|{}|{:.1f} mins|\n".format(i+1, p, signal_value_dict[p]/60)
    text += '\n'
    return text

def get_atr_user_prompt(data):
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    historical_observation = get_historical_observation_atr(t0, data['Traffic_state_history'])
    current_observation = get_current_observation_atr(t1, data['Traffic_state_history'])
    prompt = get_advance_traffic_reasoning_prompt(t0, t1, historical_observation, current_observation)
    return prompt

def get_atr_user_prompt_adaptive(data, complex_level):
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    historical_observation = get_historical_observation_atr(t0, data['Traffic_state_history'])
    current_observation = get_current_observation_atr(t1, data['Traffic_state_history'])
    prompt = get_advance_traffic_reasoning_prompt_adaptive(t0, t1, historical_observation, current_observation, complex_level)
    return prompt

def get_ra_user_prompt(data, atr_results):
    t0 = data['Timestep']-1
    t1 = data['Timestep']
    current_observation_text = get_current_observation_ra(t1, data['Traffic_state_history'])
    local_signal_priority = get_local_signal_priority_text(data['Signal_Rank'])
    prompt = get_reactive_action_prompt(t0, t1, advance_traffic_reasoning_results=atr_results, current_observation=current_observation_text, local_signal_priority=local_signal_priority)
    return prompt

def critic_neighbor_check(data):
    neighboring_critic_exist = False
    if len(data['Traffic_state_history']) > 1:
        traffic_state = data['Traffic_state_history'][-2]
        for lane in traffic_state:
            if "upstream" in lane or "downstream" in lane:
                if traffic_state[lane]['Occupancy (%)'] >= 40:
                    neighboring_critic_exist = True
                    break

    return neighboring_critic_exist

def get_atr_results_by_LLM(data):
    neighboring_critic_exist = critic_neighbor_check(data)

    if neighboring_critic_exist:
        complex_level = 'complex'
        atr_prompt = get_atr_user_prompt_adaptive(data, complex_level='complex')
    else:
        complex_level = 'easy'
        atr_prompt = get_atr_user_prompt_adaptive(data, complex_level='easy')
    atr_results = LLM.ask(atr_prompt, system_prompt=SYSTEM_PROMPT)

    return atr_results, complex_level

def get_ra_results_by_LLM(ra_user_prompt):
    ra_results = LLM.ask(ra_user_prompt, system_prompt=SYSTEM_PROMPT)
    return ra_results

def _request(cfg,task,index,function,*args):
    start=time.perf_counter();before=LLM.requests_count;usage=dict(LLM.usage)
    try:return function(*args)
    finally:
        attempts=LLM.requests_count-before
        append_jsonl(cfg.out/'logs/teacher_requests.jsonl',{'record_index':index,'task':task,
            'model':cfg['TEACHER_MODEL'],'attempts':attempts,'retries':max(0,attempts-1),
            'latency_s':time.perf_counter()-start,'usage':{k:LLM.usage.get(k,0)-usage.get(k,0) for k in LLM.usage}})


@exclusive_stage
def synthesize(cfg):
    global LLM
    if not cfg.run_dir:raise ValueError('Choose a training run with --run-dir')
    LLM = GPT_model(model=cfg['TEACHER_MODEL'],key=cfg.secret(),
        url=cfg['TEACHER_BASE_URL'].rstrip('/')+'/chat/completions',
        timeout=cfg['TEACHER_TIMEOUT_SECONDS'],max_attempts=cfg['TEACHER_MAX_ATTEMPTS'],
        use_env_proxy=cfg['TEACHER_USE_ENV_PROXY'],llm_params={'temperature':cfg['TEACHER_TEMPERATURE'],
        'max_tokens':cfg['TEACHER_MAX_TOKENS'],'enable_thinking':cfg['TEACHER_ENABLE_THINKING']})
    init_data_list=read_json(cfg.path('TRAFFIC_SAMPLE_PATH'))
    # Published filtering and shuffled-prefix selection. No custom ID, sorting,
    # deduplication, format adapter, or replacement traffic eligibility rule.
    data_list=[]
    for data in init_data_list:
        if data['Timestep']==0 or len(data['Current_Observation']['non_empty_lanes'])==0:continue
        data_list.append(data)
    random.Random(cfg['TEACHER_RANDOM_SEED']).shuffle(data_list)
    maximum=cfg['TEACHER_MAX_SAMPLES']
    if maximum and len(data_list)>maximum:
        atomic_json(cfg.data_dir/'traffic_samples/env_rl_data.json',data_list[maximum:])
        data_list=data_list[:maximum]
    if len(data_list)<2:raise ValueError('At least two eligible traffic records are required for SFT train/validation')
    output=cfg.path('TEACHER_OUTPUT_PATH')
    total_data=read_json(output) if output.exists() else []
    if len(total_data)>len(data_list) or any(row['data']!=data_list[i] for i,row in enumerate(total_data)):
        raise ValueError('Existing teacher prefix belongs to different input data')
    resumed=len(total_data)
    for i,data in enumerate(data_list):
        if i<resumed:continue
        atr_user_prompt=get_atr_user_prompt(data)
        atr_results,complex_level=_request(cfg,'atr',i,get_atr_results_by_LLM,data)
        ra_user_prompt=get_ra_user_prompt(data,atr_results)
        ra_results=_request(cfg,'ra',i,get_ra_results_by_LLM,ra_user_prompt)
        total_data.append({'atr_user_prompt':atr_user_prompt,'atr_output':atr_results,
            'ra_user_prompt':ra_user_prompt,'ra_output':ra_results,'data':data,'complex_level':complex_level})
        atomic_json(output,total_data)
        print(f'Teacher completed {len(total_data)}/{len(data_list)}',flush=True)
    cfg.report('teacher_report.json',{'input_records':len(init_data_list),'selected':len(data_list),
        'completed':len(total_data),'resumed_records':resumed,'requests_this_run':LLM.requests_count,'usage_this_run':LLM.usage})
    return total_data


if __name__=='__main__':
    args=config_args(__doc__).parse_args()
    synthesize(TrainingConfig(args.env_file,config_file=args.config_file,run_dir=args.run_dir))
