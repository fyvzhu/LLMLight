from pathlib import Path
from utils.my_utils import *
from utils.utils import *
from utils.LLMs import *
from models.CoLLMLightAgent import *
import os
import time
import numpy as np
from utils.cityflow_env import CityFlowEnv
from utils.prompts import SYSTEM_PROMPT

from tqdm import tqdm
from copy import deepcopy
import re

def extract_json(response):
    try:
        match = re.search(r"({.*})", response.strip(), re.DOTALL)
        if match:
            json_data = match.group(1)  # 提取 JSON 正文部分
            data = json.loads(json_data)
            return data
        else:
            print("No JSON data found in the response.")
            return None
        
    except json.JSONDecodeError as e:
        return None
        # raise RuntimeError(f"Parse error: {e}")

class CoLLMLight:
    # effective and efficient communication - LLM adaptive Communication module, activated by a heuristic rule
    # efficient decision making -  3-level schema
    # complex reasoning -> LLM slow thinking
    # Evolution: LLM adjust the params of each intersection according to memory. PEND
    # Memory: history data (interactions) colloct for each intersection
    # visualization
    # agent build
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow, my_config):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.env = None
        self.roadnet = roadnet
        self.dataset_name = my_config['dataset']
        self.intersection_list = None
        self.num_col = dic_traffic_env_conf['NUM_COL']
        self.num_row = dic_traffic_env_conf['NUM_ROW']
        self.trafficflow = trafficflow
        self.my_config = my_config
        self.memo = my_config['memo']
        self.history_data = None
        
        self.signal_time = 30
        self.seg_num = 10
        self.LLM_rate = 1 # how many inter use LLM to check congestion
        self.default_release_range = 9
        self.LLM_start_timepoint = 0
        self.max_boundary_release_mode = False
        self.meet_max = False
        self.params = None
        self.ignore_occupancy_threshold = my_config['communication_threshold']
        self.h_w_size = my_config['h_w_size']
        self.car_spacing = 9 
        self.signal_list = list(four_phase_list.keys())
        self.debug_wollm = my_config['debug_wollm']
        self.disable_psr = my_config['disable_psr']
        self.future_inference_depth = 1
        self.history_states = []
        self.initialize()
        
    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)
        os.makedirs(self.dic_path["PATH_TO_LOG"],exist_ok=True)
        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_LOG"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.intersection_list = list(self.env.intersection_dict.keys())
        self.history_data_init()
        # self.boundary_intersections = self.init_boundary_intersections()
        self.llm_init()
        self.agent_init()
        self.release_data = []
    
    def agent_init(self):
        traffic_env_conf = {}
        traffic_env_conf['name'] = self.my_config['dataset']
        traffic_env_conf['size'] = (self.num_row, self.num_col)
        traffic_env_conf['signal_list'] = list(four_phase_list.keys())
        traffic_env_conf['lane_list'] = location_all_direction_dict #T and L , R
        traffic_env_conf['signal_time'] = self.signal_time
        traffic_env_conf['intersection_list'] = self.intersection_list
        self.agent_intersection_list = []
        for i, inter_name in enumerate(self.intersection_list):
            agent_conf = {}
            agent_conf['inter_name'] = inter_name
            agent_conf['inter_id'] = i
            # agent_conf['boundary'] = True if i in self.boundary_intersections['id_list'] else False
            agent_conf['h_w_size'] = self.h_w_size
            agent = IntersectionAgent(agent_conf, traffic_env_conf, self.LLM)
            agent.neighbor_list = self.get_neighbor_list(i)
            self.agent_intersection_list.append(agent)
        # self.global_control_agent = GlobalAgent(traffic_env_conf, self.LLM)
    

    def llm_init(self):
        # congestion_intersection
        llm_params = {
            "max_tokens": self.my_config['max_tokens'],"seed":self.my_config.get('seed',42)}
        model = self.my_config['llm_path']
        model_type = self.my_config['llm_type']
        if model_type == 'gpt':
            self.LLM = GPT_model(model=model)
        elif model_type == 'llama':
            self.LLM = LLAMA_model(llm_params = llm_params, model=model, port=self.my_config['port'])
    
        self.LLM.generation_seed=self.my_config.get('seed',42)
        if self.my_config.get('evaluation_dir'):
            from utils.inference_logging import ObservedLLM
            self.LLM=ObservedLLM(self.LLM,self.my_config['evaluation_dir'])

    def observation_snapshot(self, ids):
        from utils.inference_logging import neighbor_lane
        snapshots=[]
        for inter_id in ids:
            agent=self.agent_intersection_list[inter_id]
            use_current=agent.syn_psr
            observation=deepcopy(agent.traffic_state_log[-self.h_w_size:] if use_current else agent.traffic_state_log[-self.h_w_size-1:-1])
            current=observation[-1] if observation else {}
            own=set();neighbor=set();neighbor_ids=set()
            past=self.history_states[-1] if use_current else (self.history_states[-2] if len(self.history_states)>1 else None)
            own_count=0;neighbor_count=0
            for lane,item in current.items():
                if lane in ('Signal','Summary') or not isinstance(item,dict):continue
                count=item.get('Queued Cars',0)+item.get('Moving Cars',0)
                if neighbor_lane(lane) is not None:
                    index,key=neighbor_lane(lane);neighbor_ids.add(index);neighbor_count+=count
                    if past is not None and isinstance(index,int) and 0<=index<len(past):neighbor.update(past[index].get(key,{}).get('veh2cell',{}))
                else:
                    own_count+=count
                    if past is not None:own.update(past[inter_id].get(lane,{}).get('veh2cell',{}))
            snapshots.append({'own_vehicle_count':own_count,'neighbor_vehicle_count':neighbor_count,
                'observed_vehicle_union_count':len(own|neighbor) if past is not None else None,
                'vehicle_count_scope':'lanes in latest SR historical observation; unique vehicle IDs when available',
                'observed_neighbor_ids':list(neighbor_ids)})
        return snapshots

    def history_data_init(self):
        self.history_data = {}
        self.history_data["release_range"] = {lane:{} for lane in location_direction_dict}
        self.history_data["release_car_num"] = [] # the total number of car release by the lane/signal in each step
        self.history_data["input_car_num"] = []
        self.history_data['effective_release_log'] = {}
        self.history_data['perf'] = {}
        self.history_data['perf']['AWT'] = []
        self.history_data['perf']['AQL'] = []
        self.history_data['perf']['ATT'] = []

        self.history_data['llm_resulst'] = {}
        for name in self.intersection_list:
            self.history_data['llm_resulst'][name] = []
            # self.history_data['llm_resulst'][name]['prompt'] = []
            # self.history_data['llm_resulst'][name]['response'] = []
        
        self.history_data["car_num_inside"] = {}
        self.history_data["car_num_inside"]['waiting'] = []
        self.history_data["car_num_inside"]['running'] = []
        self.history_data["car_num_inside"]['total'] = []
        self.history_data["car_num_inside"]['r_i'] = [] # count according to release and input of boundary inter
        self.history_data['car_num_outside'] = {}
        self.history_data["car_num_outside"]['waiting'] = []
        self.history_data["car_num_outside"]['running'] = []
        self.history_data["car_num_outside"]['total'] = []
        self.history_data["veh_num"] = []
        self.history_data["ql_num"] = []
        self.history_data['reject_num'] = {}
        self.history_data['sum_wait_time'] = 0
        

        self.history_data["car_num_out_of_lane"] = []
        self.history_data["avg_ql_cells"] = []


        self.history_data['boundary'] = {}
        self.history_data['boundary']['release_num'] = []
        self.history_data['boundary']['input_num'] = []
        self.history_data['boundary']['r_i_dif'] = []
        self.history_data['boundary']['max_release'] = []
        self.history_data['boundary']['max_input'] = []
        self.history_data['boundary']['max_r_i_dif'] = []
        self.history_data['boundary']['sum_release'] = []
        self.history_data['boundary']['sum_input'] = []
        self.history_data['boundary']['sum_r_i_dif'] = []
        self.history_data['boundary']['min_release'] = []
        self.history_data['boundary']['min_input'] = []
        self.history_data['boundary']['min_r_i_dif'] = []
        

        #debug
        self.history_data['veh_log'] = {}
        self.history_data['veh_log']['outside_list'] = []
        self.history_data['veh_log']['inside_list'] = []

    def init_boundary_intersections(self):
        boundary_data = {}
        boundary_name_list = []
        boundary_id_list = []
        
        for row_id in range(1, self.num_row+1):
            name_1 = 'intersection_'+'1_'+ str(row_id)
            name_2 = 'intersection_'+ str(self.num_col) +'_'+ str(row_id)
            boundary_name_list.append(name_1)
            boundary_id_list.append(self.intersection_list.index(name_1))
            boundary_name_list.append(name_2)
            boundary_id_list.append(self.intersection_list.index(name_2))
        for col_id in range(2, self.num_col):
            name_1 = 'intersection_'+ str(col_id) +'_1'
            name_2 = 'intersection_'+ str(col_id) +'_' + str(self.num_row)
            boundary_name_list.append(name_1)
            boundary_id_list.append(self.intersection_list.index(name_1))
            boundary_name_list.append(name_2)
            boundary_id_list.append(self.intersection_list.index(name_2))

        # 将 boundary_id_list 和 boundary_name_list 打包在一起，排序后再解包
        sorted_boundary = sorted(zip(boundary_id_list, boundary_name_list))

        # 解包排序后的 boundary_id_list 和 boundary_name_list
        boundary_id_list, boundary_name_list = zip(*sorted_boundary)

        # 转换成列表
        boundary_id_list = list(boundary_id_list)
        boundary_name_list = list(boundary_name_list)
        
        boundary_data['id_list'] = boundary_id_list
        boundary_data['name_list'] = boundary_name_list

        return boundary_data
        
    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        total_atr_time = 0
        total_ra_time = 0
        total_inference_ct = 0
        self.llm_fail_ct = 0
        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = [] 
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds
        
        start_time = time.time()
        # state_action_log = [[] for _ in range(len(state))]
        current_states, current_outputs = self.process_state(state)
        self.current_states = current_states
        last_veh_num = 0
        last_action_list = [None]*len(self.current_states)
        self.veh_last_release_mark = {}
        avg_infer_speed = []
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            total_step_num = int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME'])
            if done or current_time >= total_run_cnt:
                break
            # update states
            veh_release_share = {}
            veh_input_share = {}
            self.history_states.append(deepcopy(current_states))
            for i, state in enumerate(current_states):
                self.agent_intersection_list[i].update_state(state, last_action_list[i])
                # up_down_stream_view = self.get_up_down_stream_view(i, up_down_stream_relation)
                veh_release_share[i] = self.agent_intersection_list[i].veh_release_data
                veh_input_share[i] = self.agent_intersection_list[i].veh_input_data

            # update_traffic_memories and up down stream view
            self.update_veh_release_mark(veh_release_share)
            lane2upstream_data, lane2downstream_data = self.find_trajectory(veh_input_share)


            for i, state in enumerate(current_states):
                self.agent_intersection_list[i].traffic_memory_update(lane2upstream_data, lane2downstream_data)
                up_down_stream_view = self.get_up_down_stream_view_from_trajectory(i)
                self.agent_intersection_list[i].update_up_down_stream_view(up_down_stream_view)
                if step_num > 0:
                    traffic_state_updown_stream = self.get_up_down_stream_traffic_state(i)
                    self.agent_intersection_list[i].update_traffic_state_updown_stream(traffic_state_updown_stream)

            action_list = [None] * len(self.agent_intersection_list)
            
            ## decision making
            decision_making_intersection_id_list = []
            for inter_id, inter_agent in enumerate(self.agent_intersection_list):
                state = current_states[inter_id]
                car_exist = 0
                for lane in location_direction_dict:
                    car_exist += state[lane]['occupancy']
                effective_range_list = self.estimate_effective_range_new(state)
                if car_exist == 0:
                    signal = self.agent_intersection_list[inter_id].select_signal_default(effective_range_list)
                    action_list[inter_id] = action2code(signal)
                else:
                    decision_making_intersection_id_list.append(inter_id)
                    
            inference_num = len(decision_making_intersection_id_list)
            if len(decision_making_intersection_id_list) > 0:
                print("Time step {}: {} non-empty intersections decision making ...".format(step_num, inference_num))
                total_inference_ct += inference_num
                action_list, decision_making_messages_list, atr_score_list, time_cost = self.Asynchronous_Reasoning_and_Acting(action_list, decision_making_intersection_id_list, step_num)
                total_atr_time += time_cost['atr']
                total_ra_time += time_cost['ra']
                # print time cost
                # print('Batch size: {}. Time cost: ATR {:.2f}s, RA {:.2f}s'.format(inference_num, time_cost['atr'], time_cost['ra']))

            # be sure all action is not None
            llm_fail_num = len([x for x in action_list if x == None])
            self.llm_fail_ct += llm_fail_num
            print('LLM fail num: {}'.format(llm_fail_num))
            for i, action in enumerate(action_list):
                if action == None:
                    action_list[i] = action2code('ETWT')

            # save LLM messages and atr_score
            for inter_id, inter_name in enumerate(self.intersection_list):
                prompt_response_dict = {}
                prompt_response_dict['step'] = step_num
                if inter_id in decision_making_intersection_id_list:
                    prompt_response_dict['decision_making_messages'] = decision_making_messages_list[decision_making_intersection_id_list.index(inter_id)]
                    prompt_response_dict['atr_score'] = atr_score_list[decision_making_intersection_id_list.index(inter_id)]
                    self.history_data['llm_resulst'][inter_name].append(prompt_response_dict)
            
            next_state, _, done, _ = self.env.step(action_list)
            next_states, next_outputs = self.process_state(next_state)
            last_action_list = action_list[:]
            
            # self.update_history_data(current_states, next_states, current_outputs, next_outputs, action_list)
            # global_indicator = {}
            # global_indicator['release'] = self.history_data['release_car_num'][-1]
            # global_indicator['input'] = self.history_data['input_car_num'][-1]
            # self.global_control_agent.update_state(global_indicator)

            # print("all_lane_current_release: {}, max_release: {}".format(self.history_data['release_car_num'][-1], max(self.history_data['release_car_num'])))
            # print("all_lane_current_input: {}, max_input: {}".format(self.history_data['input_car_num'][-1], max(self.history_data['input_car_num'])))
            # # print("current_car_num: {}, max_car_num: {}".format(self.history_data['car_num'][-1], max(self.history_data['car_num'])))
            
            # print("current_car_num_ool: {}, max_car_num_ool: {}".format(self.history_data['car_num_out_of_lane'][-1], max(self.history_data['car_num_out_of_lane'])))
            
            print("total llm fail num: {}".format(self.llm_fail_ct))
            if self.my_config['llm_type'] == 'gpt':
                print("gpt consume: {}".format(self.LLM.total_consume))
            
            rewards = self.get_norm_reward(next_state)  # my reward

            current_time = self.env.get_current_time()  # in seconds
            # state = next_state
            current_states = next_states
            self.current_states = current_states
            current_outputs = next_outputs
            # calculate logger results
            total_reward += sum(rewards)
            queue_length_inter = []
            veh_num = []

            for inter in self.env.list_intersection:
                veh_num.append(sum(inter.dic_feature['lane_num_vehicle']))
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter)) 
            # queue_length_inter = np.array(queue_length_inter)
            # ql_num_matrix = queue_length_inter.reshape(self.num_col, self.num_row)
            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)
            self.history_data['sum_wait_time'] = int(sum(waiting_times))
            veh_num = np.sum(veh_num)
            
            ## perf at this point
            awt = round(np.mean(waiting_time_episode),3)
            aql = round(np.mean(queue_length_episode),3)
            att = self.calc_avg_travel_time()
            att = round(att,3)
            throughput = self.calc_throughput()
            print("Queuing Vehicles:", np.sum(queue_length_inter), "Veh num: ", veh_num, "new Vehs: ", veh_num - last_veh_num, "AWT: ", awt, "AQL:", aql, "ATT:", att, 'Throughput:', throughput)
            if self.my_config.get('evaluation_dir'):
                from utils.experiment import append_jsonl
                append_jsonl(Path(self.my_config['evaluation_dir'])/'metrics/traffic.jsonl',
                    {'step':step_num,'simulation_time':current_time,'queue_network_sum':sum(queue_length_inter),
                     'queue_by_intersection':queue_length_inter,
                     'waiting_vehicle_count':len(waiting_times),'waiting_time_sum':sum(waiting_times),
                     'AWT':awt,'AQL':aql,'ATT':att,'throughput':throughput,
                     'definitions':'original CoLLMLight: AQL=time-mean network queue sum; AWT=time-mean stopped-vehicle mean wait; ATT=sum intersection residence per vehicle'})
            self.history_data['perf']['AWT'].append(awt)
            self.history_data['perf']['AQL'].append(aql)
            self.history_data['perf']['ATT'].append(att)
            # print("ql statistics: mean: {}, var: {}, min: {}, max: {}, 0ql_num: {}, Max_Avg_ql_cells: {}, Avg_ql_cells: {}".format(round(np.mean(queue_length_inter),4), round(np.var(queue_length_inter),4), round(np.min(queue_length_inter),4), round(np.max(queue_length_inter),4), (queue_length_inter==0).sum(), round(np.max(self.history_data["avg_ql_cells"][-1]),4), round(np.mean(self.history_data["avg_ql_cells"][-1]),4)))
            # boundary release and input statistics
            # self.show_boundary_data()
            # self.show_car_num()
            last_veh_num = veh_num
        total_travel_time = self.calc_avg_travel_time()
        throughput = self.calc_throughput()
        
        results = {
            "test_reward_over": total_reward,
            "test_throughput_over": throughput,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time, 
            "LLM fail num": self.llm_fail_ct,
            "total_atr_time": total_atr_time,
            "total_ra_time": total_ra_time,
            "avg_atr_time": total_atr_time/total_inference_ct if total_inference_ct else 0.0,
            "avg_ra_time": total_ra_time/total_inference_ct if total_inference_ct else 0.0,
            "total_inference_ct": total_inference_ct}
        logger.log(results)
        print('Test Round:', test_round, results)
        print('Testing time:', time.time()-start_time)

        self.env.batch_log_2()

        return results
    
    def Asynchronous_Reasoning_and_Acting(self, action_list, decision_making_intersection_id_list, step_num):
        # 1st stage Advance Traffic Reasoning  

        advance_traffic_reasoning_output_list, advance_traffic_reasoning_messages_list, atr_total_time_cost = self.advance_traffic_reasoning(decision_making_intersection_id_list, step_num)
        print("atr done: {:.2f} s".format(atr_total_time_cost))

        # 2nd stage Reactive Action  
        action_list, reactive_action_messages_list, atr_score_list, ra_total_time_cost = self.reactive_action(action_list, advance_traffic_reasoning_output_list, decision_making_intersection_id_list, step_num)
        print("ra done: {:.2f} s".format(ra_total_time_cost))
        total_messages_list = [advance_traffic_reasoning_messages_list[i] + action_message for i, action_message in enumerate(reactive_action_messages_list)]
        
        time_cost = {}
        time_cost['atr'] = atr_total_time_cost
        time_cost['ra'] = ra_total_time_cost
        return action_list, total_messages_list, atr_score_list, time_cost
    

    def advance_traffic_reasoning(self, decision_making_intersection_id_list, step_num):
        decision_making_1st_prompts_list = []
        for inter_id in decision_making_intersection_id_list:
            agent = self.agent_intersection_list[inter_id]
            decision_making_1st_prompts_list.append(agent.advance_traffic_reasoning_prompt(step_num))
        if hasattr(self.LLM,'set_context'):
            self.LLM.set_context(module='SR',step=step_num,intersections=list(decision_making_intersection_id_list),
                observations=self.observation_snapshot(decision_making_intersection_id_list))
        atr_start_time = time.time()
        if self.disable_psr:
            advance_traffic_reasoning_output_list = ["" for _ in range(len(decision_making_1st_prompts_list))]

        else:
            advance_traffic_reasoning_output_list = self.LLM.batch_ask(decision_making_1st_prompts_list, system_prompt=SYSTEM_PROMPT, disable_thinking=True)
        # 提取 </think> 标签后的内容
            advance_traffic_reasoning_output_list = [self.extract_after_think_tag(output) for output in advance_traffic_reasoning_output_list]
            

        atr_total_time_cost = time.time()-atr_start_time
        advance_traffic_reasoning_messages_list = []
        for i in range(len(decision_making_intersection_id_list)):
            messages = [{"role": "user", "content": decision_making_1st_prompts_list[i]},
                        {"role": "assistant", "content": advance_traffic_reasoning_output_list[i]}]
            advance_traffic_reasoning_messages_list.append(messages)
        return advance_traffic_reasoning_output_list, advance_traffic_reasoning_messages_list, atr_total_time_cost

    def reactive_action(self, action_list, advance_traffic_reasoning_output_list, decision_making_intersection_id_list, step_num):
        decision_making_2st_prompts_list = []
        for i, inter_id in enumerate(decision_making_intersection_id_list):
            agent = self.agent_intersection_list[inter_id]
            decision_making_2st_prompts_list.append(agent.reactive_action_prompt(advance_traffic_reasoning_output_list[i], step_num)+getattr(agent,'_unavailable_note',''))
        if hasattr(self.LLM,'set_context'):
            self.LLM.set_context(module='RD',step=step_num,intersections=list(decision_making_intersection_id_list))
        ra_start_time = time.time()
        reactive_action_output_list = self.LLM.batch_ask(decision_making_2st_prompts_list, system_prompt=SYSTEM_PROMPT, disable_thinking=True)
        ra_total_time_cost = time.time()-ra_start_time
        
        reactive_action_messages_list = []
        atr_score_list = []
        for i, inter_id in enumerate(decision_making_intersection_id_list):
            messages = [{"role": "user", "content": decision_making_2st_prompts_list[i]},
                        {"role": "assistant", "content": reactive_action_output_list[i]}]
            reactive_action_messages_list.append(messages)
            results = extract_json_reactive_action_results(reactive_action_output_list[i])
            if results['signal'] in self.signal_list:
                action_list[inter_id] = self.signal_list.index(results['signal'])
            atr_score_list.append(results['used_atr'])

            
        return action_list, reactive_action_messages_list, atr_score_list, ra_total_time_cost
    
    def extract_after_think_tag(self, text):
        """
        提取 </think> 标签后面的内容，如果没有该标签则返回原始字符串。
        """
        tag = "</think>"
        if tag in text:
            return text.split(tag, 1)[1].strip()
        return text
    

    
    

    
    # communicate_lanes = self.adaptive_communication(communicate_data_check_prompt_list, communicate_data_location_list)
    def adaptive_communication(self, communicate_data_check_prompt_list, communicate_data_location_list):
        communicate_lanes = []
        if len(communicate_data_check_prompt_list) > 0:
            llm_responses = self.LLM.batch_ask(communicate_data_check_prompt_list)
            for i, response in enumerate(llm_responses):
                answer = extract_answer_by_tag("answer", response)
                if answer:
                    if answer.lower() == 'no':
                        continue
                elif 'yes' not in response.lower():
                        continue
                else:
                    communicate_lanes.append(communicate_data_location_list[i])
        return communicate_lanes
    
    def multi_level_decision_making_batch(self, action_list, decision_making_batch_prompt_list, decision_making_intersection_id_list):
        system_prompt = "You are a traffic signal expert responsible for managing a four-way intersection. Your primary goal is to assess the current coordination level and implement appropriate signal selection strategies. Aim to optimize traffic flow and safety, considering not only your intersection but also the impacts on adjacent upstream and downstream intersections.\n"
        decision_making_response_list = self.LLM.batch_ask(decision_making_batch_prompt_list, system_prompt)
        coordination_distribution = {}
        coordination_distribution['NO'] = 0
        coordination_distribution['Simple'] = 0
        coordination_distribution['Complex'] = 0
        

        for i, inter_id in enumerate(decision_making_intersection_id_list):
            answer_dict = extract_json(decision_making_response_list[i])
            if answer_dict:
                if self.think_mode in ['quick', 'mild', 'slow']:
                    if 'answer' in answer_dict:
                        signal = answer_dict['answer']
                        if signal in self.signal_list:
                            action_list[inter_id] = action2code(signal)
                else:
                    if 'phase2' in answer_dict:
                        if 'answer' in answer_dict['phase2']:
                            signal = answer_dict['phase2']['answer']
                            if signal in self.signal_list:
                                action_list[inter_id] = action2code(signal)
                    if 'phase1' in answer_dict:
                        if 'answer' in answer_dict['phase1']:
                            coordination = answer_dict['phase1']['answer']
                            if 'no' in coordination.lower():
                                coordination_distribution['NO'] += 1
                            elif 'simple' in coordination.lower():
                                coordination_distribution['Simple'] += 1
                            elif 'complex' in coordination.lower():
                                coordination_distribution['Complex'] += 1

        ## normalize coordination distribution
        sum_coordination = coordination_distribution['NO'] + coordination_distribution['Simple'] + coordination_distribution['Complex']
        if sum_coordination == 0:
            sum_coordination = 1
        coordination_distribution['NO'] = coordination_distribution['NO'] / sum_coordination * 100
        coordination_distribution['Simple'] = coordination_distribution['Simple'] / sum_coordination * 100
        coordination_distribution['Complex'] = coordination_distribution['Complex'] / sum_coordination * 100
        return action_list, decision_making_response_list, coordination_distribution

    
    def fast_thinking_batch(self, action_list, first_batch_prompt_list, first_thinking_intersection_id_list):
        fast_thinking_response = []
        medium_thinking_intersection_id_list = []
        slow_thinking_intersection_id_list = []
        system_prompt = "You are a traffic signal expert at a four-way intersection with 12 lanes: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR]. Each lane is labeled by direction and movement: N for north, S for south, E for east, W for west, L for left turn, T for through, and R for right turn. For instance, ET stands for the East Through lane, where traffic moves straight ahead from east to west. WL is the West Left-turn lane, where traffic turns left from west to north. Right turns are always allowed. There are four signal options: [ETWT, NTST, ELWL, NLSL]. For example, ETWT indicates the release of both the ET and WT lanes. Your goal is to classify the given traffic observation into one of three coordination scenarios based on a detailed analysis: no-coordination, simple-coordination, and complex-coordination.\n"
        # system_prompt = "You are a traffic signal controller at a four-way intersection with 12 lanes: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR]. Each lane is labeled by direction and movement: N for north, S for south, E for east, W for west, L for left turn, T for through, and R for right turn. For instance, ET stands for the East Through lane, where traffic moves straight ahead from east to west. WL is the West Left-turn lane, where traffic turns left from west to north. Right turns are always allowed. There are four signal options: [ETWT, NTST, ELWL, NLSL]. For example, ETWT indicates the release of both the ET and WT lanes. Your goal is to optimize traffic flow and coordinate with nearby intersections.\n"
        if len(first_batch_prompt_list) >= 100:
            for i in range(0, len(first_batch_prompt_list), 100):
                batch_response = self.LLM.batch_ask(first_batch_prompt_list[i:i+100], system_prompt)
                fast_thinking_response.extend(batch_response)
        else:
            fast_thinking_response = self.LLM.batch_ask(first_batch_prompt_list, system_prompt)
        
        for i, response in enumerate(fast_thinking_response):
            answer = extract_answer_by_tag("answer", response)
            inter_id = first_thinking_intersection_id_list[i]
            if answer.lower() == 'complex':
                slow_thinking_intersection_id_list.append(inter_id)
                # self.llm_fail_ct += 1
            elif answer.lower() == 'simple':
                medium_thinking_intersection_id_list.append(inter_id)
            else:
                state = self.current_states[inter_id]
                effective_range_list = self.estimate_effective_range_new(state)
                signal = self.agent_intersection_list[inter_id].select_signal_default(effective_range_list)
                action_list[inter_id] = action2code(signal)

        return action_list, medium_thinking_intersection_id_list, slow_thinking_intersection_id_list, fast_thinking_response
    
    def medium_thinking_batch(self, action_list, medium_batch_prompt_list, medium_thinking_intersection_id_list):
        medium_thinking_response = []
        system_prompt = "You are a traffic signal controller at a four-way intersection with 12 lanes: [NL, NT, NR, SL, ST, SR, EL, ET, ER, WL, WT, WR]. Each lane is labeled by direction and movement: N for north, S for south, E for east, W for west, L for left turn, T for through, and R for right turn. For instance, ET stands for the East Through lane, where traffic moves straight ahead from east to west. WL is the West Left-turn lane, where traffic turns left from west to north. Right turns are always allowed. There are four signal options: [ETWT, NTST, ELWL, NLSL]. For example, ETWT indicates the release of both the ET and WT lanes. Your goal is to optimize traffic flow and coordinate with nearby intersections to minimize wait times and queues by selecting the best signal phases based on current conditions.\n"
        if len(medium_batch_prompt_list) >= 100:
            for i in range(0, len(medium_batch_prompt_list), 100):
                batch_response = self.LLM.batch_ask(medium_batch_prompt_list[i:i+100], system_prompt)
                medium_thinking_response.extend(batch_response)
        else:
            medium_thinking_response = self.LLM.batch_ask(medium_batch_prompt_list, system_prompt)
        
        for i, response in enumerate(medium_thinking_response):
            answer = extract_answer_by_tag("answer", response)
            inter_id = medium_thinking_intersection_id_list[i]
            if answer in self.signal_list:
                action_list[inter_id] = action2code(answer)

        return action_list, medium_thinking_response

    

    




    


        











    
    def get_up_down_stream_view_from_trajectory(self, inter_id):
        view = {}
        view['exist'] = False
        no_empty_lanes = self.agent_intersection_list[inter_id].no_empty_lanes
        #
        for lane in no_empty_lanes:
            view[lane] = {}
            upstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['upstream_lanes']
            downstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['downstream_lanes']
            if len(upstream_lanes) > 0:
                view[lane]['upstream'] = {}
                for upstream_id, up_lane in upstream_lanes:
                    if self.current_states[upstream_id][up_lane]['occupancy'] > self.ignore_occupancy_threshold:
                    # if (upstream_id, up_lane) in self.communicate_lanes:
                        view[lane]['upstream'][up_lane] = self.current_states[upstream_id][up_lane]
                        view[lane]['upstream'][up_lane]['inter_id'] = upstream_id
                        view['exist'] = True 
                
            if len(downstream_lanes) > 0:
                view[lane]['downstream'] = {}
                for downstream_id, down_lane in downstream_lanes:
                    if self.current_states[downstream_id][down_lane]['occupancy'] > self.ignore_occupancy_threshold:
                    # if (downstream_id, down_lane) in self.communicate_lanes:
                        view[lane]['downstream'][down_lane] = self.current_states[downstream_id][down_lane]
                        view[lane]['downstream'][down_lane]['inter_id'] = downstream_id
                        view['exist'] = True

        return view 
    def get_up_down_stream_traffic_state(self, inter_id):
        traffic_state_updown_stream = {}
        for lane in location_direction_dict:
            upstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['upstream_lanes']
            downstream_lanes = self.agent_intersection_list[inter_id].traffic_input_memory[lane]['downstream_lanes']
            if len(upstream_lanes) > 0:
                for upstream_id, up_lane in upstream_lanes:
                    if self.current_states[upstream_id][up_lane]['occupancy'] > self.ignore_occupancy_threshold:
                    # if (upstream_id, up_lane) in self.communicate_lanes:
                        name = f"{lane}'s upstream lane ({upstream_id}, {up_lane})"
                        last_state = self.history_states[-2][upstream_id][up_lane]
                        current_state = self.history_states[-1][upstream_id][up_lane]
                        traffic_state_updown_stream[name] = self.get_lane_traffic_state(last_state, current_state)

            if len(downstream_lanes) > 0:
                for downstream_id, down_lane in downstream_lanes:
                    if self.current_states[downstream_id][down_lane]['occupancy'] > self.ignore_occupancy_threshold:
                    # if (downstream_id, down_lane) in self.communicate_lanes:
                        name = f"{lane}'s downstream lane ({downstream_id}, {down_lane})"
                        last_state = self.history_states[-2][downstream_id][down_lane]
                        current_state = self.history_states[-1][downstream_id][down_lane]
                        traffic_state_updown_stream[name] = self.get_lane_traffic_state(last_state, current_state)

        return traffic_state_updown_stream
    def get_lane_traffic_state(self, last_state, current_state):
        lane_traffic_state = {}
        cars_before = list(last_state['veh2cell'].keys())
        cars_current = list(current_state['veh2cell'].keys())
        cars_output = [veh for veh in cars_before if veh not in cars_current]
        cars_input = [veh for veh in cars_current if veh not in cars_before]
        lane_traffic_state['Cars Input'] = len(cars_input)
        lane_traffic_state['Cars Output'] = len(cars_output)
        queue_before = last_state['queue_len']
        queue_current = current_state['queue_len']
        queue_diff = queue_current - queue_before
        lane_traffic_state['Queued Cars Change'] = queue_diff
        lane_traffic_state['Queued Cars'] = queue_current
        moving_before = sum(last_state['cells'])
        moving_current = sum(current_state['cells'])
        moving_diff = moving_current - moving_before
        lane_traffic_state['Moving Cars Change'] = moving_diff
        lane_traffic_state['Moving Cars'] = moving_current
        avg_wait_time_before = last_state['avg_wait_time']
        avg_wait_time_before = round(avg_wait_time_before/60, 2)
        avg_wait_time_current = current_state['avg_wait_time']
        avg_wait_time_current = round(avg_wait_time_current/60, 2)
        avg_wait_time_diff = avg_wait_time_current - avg_wait_time_before
        lane_traffic_state['Average Waiting Time Change (mins)'] = round(avg_wait_time_diff,2)
        lane_traffic_state['Average Waiting Time (mins)'] = avg_wait_time_current
        occupancy_before = last_state['occupancy']
        occupancy_before = round(occupancy_before*100, 2)
        occupancy_current = current_state['occupancy']
        occupancy_current = round(occupancy_current*100, 2)
        occupancy_diff = occupancy_current - occupancy_before
        lane_traffic_state['Occupancy Change (%)'] = round(occupancy_diff,2)
        lane_traffic_state['Occupancy (%)'] = occupancy_current

        return lane_traffic_state
    
    def get_up_down_stream_view_space(self, inter_id, up_down_stream_relation):
        view = {}
        view['exist'] = False
        no_empty_lanes = self.agent_intersection_list[inter_id].no_empty_lanes
        if len(no_empty_lanes) <= 1:
            return view
        elif len(no_empty_lanes) == 2:
            if set(no_empty_lanes) in [set(['WT', 'ET']), set(['EL', 'WL']), set(['NT', 'ST']), set(['SL', 'NL'])]:
                return view

        for lane in no_empty_lanes:
            view[lane] = {}
            upstream_id = up_down_stream_relation[lane]['upstream_id']
            upstream_location = up_down_stream_relation[lane]['upstream_location']
            downstream_id = up_down_stream_relation[lane]['downstream_id']
            downstream_location = up_down_stream_relation[lane]['downstream_location']
            if up_down_stream_relation[lane]['upstream_id'] is not None:
                view[lane]['upstream'] = {}
                for up_lane in up_down_stream_relation[lane]['upstream_lanes']:
                    if self.current_states[upstream_id][up_lane]['occupancy'] > self.ignore_occupancy_threshold:
                        view[lane]['upstream'][up_lane] = self.current_states[upstream_id][up_lane]
                        view[lane]['upstream'][up_lane]['location'] = upstream_location
                        view['exist'] = True     
            if up_down_stream_relation[lane]['downstream_id'] is not None:
                view[lane]['downstream'] = {}
                for down_lane in up_down_stream_relation[lane]['downstream_lanes']:
                    if self.current_states[downstream_id][down_lane]['occupancy'] > self.ignore_occupancy_threshold:
                        view[lane]['downstream'][down_lane] = self.current_states[downstream_id][down_lane]
                        view[lane]['downstream'][down_lane]['location'] = downstream_location
                        view['exist'] = True

        return view 
                    
                
    def update_veh_release_mark(self, veh_release_share):
        new_release_data = {}
        for inter_id in veh_release_share:
            for lane in veh_release_share[inter_id]:
                release_vehs = veh_release_share[inter_id][lane]
                for veh in release_vehs:
                    new_release_data[veh] = (inter_id, lane)
        self.veh_last_release_mark.update(new_release_data)

    def update_lane_release_results(self, veh, upstream_lane, downstream_lane):
        release_inter, release_lane = upstream_lane
        arrive_inter, arrive_lane = downstream_lane
        if veh in self.agent_intersection_list[release_inter].release_veh2memorie_index:
            release_lane, memory_idx = self.agent_intersection_list[release_inter].release_veh2memorie_index[veh]
            self.agent_intersection_list[release_inter].memories[release_lane][memory_idx]['release_results'][veh] = (arrive_inter, arrive_lane)  

    def find_trajectory(self, veh_input_share):
        lane2upstream = {}
        lane2downstream = {}
        for inter_id in veh_input_share:
            for lane in veh_input_share[inter_id]:
                downstream_lane = (inter_id, lane)
                input_vehs = veh_input_share[inter_id][lane]
                for veh in input_vehs:
                    if veh in self.veh_last_release_mark:
                        if downstream_lane not in lane2upstream:
                            lane2upstream[downstream_lane] = {}
                        upstream_lane = self.veh_last_release_mark[veh]

                        self.update_lane_release_results(veh, upstream_lane, downstream_lane)
                        
                        if upstream_lane not in lane2upstream[downstream_lane]:
                            lane2upstream[downstream_lane][upstream_lane] = 0
                        lane2upstream[downstream_lane][upstream_lane] += 1
                        if upstream_lane not in lane2downstream:
                            lane2downstream[upstream_lane] = {}

                        if downstream_lane not in lane2downstream[upstream_lane]:
                            lane2downstream[upstream_lane][downstream_lane] = 0
                        lane2downstream[upstream_lane][downstream_lane] += 1

        return lane2upstream, lane2downstream
    
    def summarize_and_update_long_distance_info(self):
        for i, inter_agent in enumerate(self.agent_intersection_list):
            long_distance_info = {}
            for lane in inter_agent.lane_list_onlyTL:
                upstream_related_lanes = {}
                downstream_related_lanes = {}
                upstream_related_lanes = self.search_related_lanes(upstream_related_lanes, i, lane, 'upstream')
                downstream_related_lanes = self.search_related_lanes(downstream_related_lanes, i, lane, 'downstream')
                long_distance_info_this_lane = self.summarize_long_distance_info(upstream_related_lanes, downstream_related_lanes)
                long_distance_info[lane] = deepcopy(long_distance_info_this_lane)
            inter_agent.update_long_distance_info(long_distance_info)
    
    def summarize_long_distance_info(self, upstream_related_lanes, downstream_related_lanes):
        long_distance_info = {}
        long_distance_info['upstream'] = {}
        long_distance_info['downstream'] = {}
        long_distance_info['exist'] = []
        if len(upstream_related_lanes) <= 1:
            long_distance_info['upstream']['exist'] = False
        else:
            total_occupancy = 0
            total_queue_num = 0
            total_waiting_time = 0
            for (inter_id, lane), state in upstream_related_lanes.items():
                total_occupancy += state['occupancy']
                total_queue_num += state['queue_car_num']
                total_waiting_time += state['wait_time']
            long_distance_info['upstream']['lane_num'] = len(upstream_related_lanes)
            long_distance_info['upstream']['average_occupancy'] = total_occupancy / len(upstream_related_lanes)
            long_distance_info['upstream']['total_queue_num'] = total_queue_num
            long_distance_info['upstream']['average_waiting_time'] = total_waiting_time / total_queue_num if total_queue_num > 0 else 0.0
            long_distance_info['upstream']['exist'] = True
            long_distance_info['exist'].append('upstream')
        
        if len(downstream_related_lanes) <= 1:
            long_distance_info['downstream']['exist'] = False
        else:
            total_occupancy = 0
            total_queue_num = 0
            total_waiting_time = 0
            for (inter_id, lane), state in downstream_related_lanes.items():
                total_occupancy += state['occupancy']
                total_queue_num += state['queue_car_num']
                total_waiting_time += state['wait_time']
            long_distance_info['downstream']['lane_num'] = len(downstream_related_lanes)
            long_distance_info['downstream']['average_occupancy'] = total_occupancy / len(downstream_related_lanes)
            long_distance_info['downstream']['total_queue_num'] = total_queue_num
            long_distance_info['downstream']['average_waiting_time'] = total_waiting_time / total_queue_num if total_queue_num > 0 else 0.0
            long_distance_info['downstream']['exist'] = True
            long_distance_info['exist'].append('downstream')

        return long_distance_info


    def search_related_lanes(self, related_lanes, i, lane, direc):
        if direc == 'upstream':
            candi_inter_lanes = self.agent_intersection_list[i].traffic_input_memory[lane]['upstream_lanes']
        elif direc == 'downstream':
            candi_inter_lanes = self.agent_intersection_list[i].traffic_input_memory[lane]['downstream_lanes']
        for inter_id, lane in candi_inter_lanes:
            if (inter_id, lane) in related_lanes:
                continue
            if self.agent_intersection_list[inter_id].current_state[lane]['occupancy'] > 0.5:
                related_lanes[(inter_id, lane)] = self.agent_intersection_list[inter_id].current_state[lane]
                related_lanes = self.search_related_lanes(related_lanes, inter_id, lane, direc)
        return related_lanes


    def find_half_occupancy_lanes_of_all_inters_with_up_down_lanes(self):
        half_occupancy_lanes_of_all_inters_with_up_down_lanes = {}
        for i, inter_agent in enumerate(self.agent_intersection_list):
            half_occupancy_lanes_of_all_inters_with_up_down_lanes[i] = {}
            for lane in inter_agent.lane_list_onlyTL:
                if inter_agent.current_state[lane]['occupancy'] > 0.5:
                    half_occupancy_lanes_of_all_inters_with_up_down_lanes[i][lane] = {}
                    half_occupancy_lanes_of_all_inters_with_up_down_lanes[i][lane]['state'] = inter_agent.current_state[lane]
                    half_occupancy_lanes_of_all_inters_with_up_down_lanes[i][lane]['upstream_inter_lanes'] = inter_agent.traffic_input_memory[lane]['upstream_lanes']
                    half_occupancy_lanes_of_all_inters_with_up_down_lanes[i][lane]['downstream_inter_lanes'] = inter_agent.traffic_input_memory[lane]['downstream_lanes']

        return half_occupancy_lanes_of_all_inters_with_up_down_lanes



    def message_passing(self):
        messages_for_all_inter = {}
        #messages_for_all_inter: {inter_id1: [{inter_id: inter_id2, side: 'North', type: 'downstream', congestion_degree:60%, congest_tree_size:3}, {...}]}

        for i in range(len(self.intersection_list)):
            messages_for_all_inter[i] = {}
        
        # (inter_id, lane), decide the upstream and downstream lane by traffic logs
        # congestion_dict: {(2, 'NL'): {upstream: [(3,'NT')], downstream: [(4,'ST')], congestion_degree:60%}
        congestion_dict = self.check_congestion() # find congestion lane and their upstream, down

        for key in congestion_dict:
            inter_id, lane = key
            congestion_degree = congestion_dict[key]['congestion_degree']
            upstream_lanes = congestion_dict[key]['upstream']
            downstream_lanes = congestion_dict[key]['downstream']
            upstream_congest_degree_list = [congestion_degree]
            upstream_congest_lane_list = [key] # include this inter itself
            upstream_queue_num_list = [self.current_states[inter_id][lane]['queue_len']]
            upstream_avg_wait_time_list = [self.current_states[inter_id][lane]['avg_wait_time']]
            upstream_congest_lane_list, upstream_congest_degree_list, upstream_queue_num_list, upstream_avg_wait_time_list = self.search_tree(key, congestion_dict, 'upstream', upstream_congest_lane_list, upstream_congest_degree_list, upstream_queue_num_list, upstream_avg_wait_time_list)
            
            downstream_congest_degree_list = [congestion_degree]
            downstream_congest_lane_list = [key]
            downstream_queue_num_list = [self.current_states[inter_id][lane]['queue_len']]
            downstream_avg_wait_time_list = [self.current_states[inter_id][lane]['avg_wait_time']]
            downstream_congest_lane_list, downstream_congest_degree_list, downstream_queue_num_list, downstream_avg_wait_time_list = self.search_tree(key, congestion_dict, 'downstream', downstream_congest_lane_list, downstream_congest_degree_list, downstream_queue_num_list, downstream_avg_wait_time_list)

            for upstream_key in upstream_lanes:
                target_id, target_lane = upstream_key
                message = {'inter_id': inter_id, 'congest_lane': lane, 'target_lane': target_lane, 'type':'downstream', 'congestion_degree': congestion_degree, 'congest_lane_list': downstream_congest_lane_list, 'congest_degree_list': downstream_congest_degree_list, 'queue_num_list': downstream_queue_num_list, 'avg_wait_time_list': downstream_avg_wait_time_list,}
                if target_lane not in messages_for_all_inter[target_id]:
                    messages_for_all_inter[target_id][target_lane] = []
                messages_for_all_inter[target_id][target_lane].append(message)

            for downstream_key in downstream_lanes:
                target_id, target_lane = downstream_key
                message = {'inter_id': inter_id, 'congest_lane': lane, 'target_lane': target_lane, 'type':'upstream', 'congestion_degree': congestion_degree, 'congest_lane_list': upstream_congest_lane_list, 'congest_degree_list': upstream_congest_degree_list, 'queue_num_list': upstream_queue_num_list, 'avg_wait_time_list': upstream_avg_wait_time_list}
                if target_lane not in messages_for_all_inter[target_id]:
                    messages_for_all_inter[target_id][target_lane] = []
                messages_for_all_inter[target_id][target_lane].append(message)
        ## aggregate same message to same lane
        for target_id in messages_for_all_inter:
            for lane in messages_for_all_inter[target_id]:
                messages = messages_for_all_inter[target_id][lane]
                congest_lane_list = {}
                congest_degree_list = {}
                queue_num_list = {}
                avg_wait_time_list = {}
                for message_type in ['upstream', 'downstream']:
                    congest_degree_list[message_type] = []
                    congest_lane_list[message_type] = []
                    avg_wait_time_list[message_type] = []
                    queue_num_list[message_type] = []
                for message in messages:
                    message_type = message['type']
                    congest_degree_list[message_type].extend(message['congest_degree_list'])
                    congest_lane_list[message_type].extend(message['congest_lane_list'])
                    queue_num_list[message_type].extend(message['queue_num_list'])
                    avg_wait_time_list[message_type].extend(message['avg_wait_time_list'])
                
                upstream_congest_num = len(congest_lane_list['upstream'])
                upstream_avg_congest = np.mean(congest_degree_list['upstream']) if upstream_congest_num > 0 else 0.0
                downstream_congest_num = len(congest_lane_list['downstream'])
                downstream_avg_congest = np.mean(congest_degree_list['downstream']) if downstream_congest_num > 0 else 0.0
                sum_queue_num_upstream = sum(queue_num_list['upstream'])
                sum_queue_num_downstream = sum(queue_num_list['downstream'])
                upstream_awt = sum([queue_num_list['upstream'][i]*avg_wait_time_list['upstream'][i] for i in range(len(queue_num_list['upstream']))])/sum_queue_num_upstream if sum_queue_num_upstream > 0 else 0.0
                downstream_awt = sum([queue_num_list['downstream'][i]*avg_wait_time_list['downstream'][i] for i in range(len(queue_num_list['downstream']))])/sum_queue_num_downstream if sum_queue_num_downstream > 0 else 0.0
                aggregated_message = {'upstream_congest_num': upstream_congest_num, 'upstream_avg_congest': upstream_avg_congest, 'downstream_congest_num': downstream_congest_num, 'downstream_avg_congest': downstream_avg_congest, 'upstream_awt': upstream_awt, 'downstream_awt': downstream_awt}
                messages_for_all_inter[target_id][lane] = aggregated_message

        return messages_for_all_inter

    def check_congestion(self):
        # congestion_dict: {(2, 'NL'): {upstream: [(3,'NT')], downstream: [(4,'ST')], congestion_degree:60%}
        congestion_dict = dict()
        for inter_id in range(len(self.intersection_list)):
            congest_data = self.agent_intersection_list[inter_id].congest_data
            congestion_dict.update(congest_data)
        return congestion_dict
            
    def search_tree(self, key, congestion_dict, direction, congest_lane_list, congest_degree_list, queue_num_list, avg_wait_time_list):
        subtree_interlanes = congestion_dict[key][direction]
        for interlane in subtree_interlanes:
            if interlane in congestion_dict and interlane not in congest_lane_list:
                inter_id, lane = interlane
                congest_lane_list.append(interlane)
                congest_degree_list.append(congestion_dict[interlane]['congestion_degree'])
                queue_num_list.append(self.current_states[inter_id][lane]['queue_len'])
                avg_wait_time_list.append(self.current_states[inter_id][lane]['avg_wait_time'])
                congest_lane_list, congest_degree_list, queue_num_list, avg_wait_time_list = self.search_tree(interlane, congestion_dict, direction, congest_lane_list, congest_degree_list, queue_num_list, avg_wait_time_list)


        return congest_lane_list, congest_degree_list, queue_num_list, avg_wait_time_list
    
    


    def calc_avg_travel_time(self):
        current_time = self.env.get_current_time()
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else current_time
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])
        return total_travel_time
    
    def calc_throughput(self):
        current_time = self.env.get_current_time()
        vehicle_travel_times = {}
        all_veh = []
        not_leave_veh = []
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                all_veh.append(veh)
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if np.isnan(leave_time):
                    not_leave_veh.append(veh)
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else current_time
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)
        leave_veh = list(set(all_veh) - set(not_leave_veh))
        throughput = len(leave_veh)

        return throughput
            
    def show_boundary_data(self):
        cbr = self.history_data['boundary']['release_num'][-1] # current_boundary_release
        cbi = self.history_data['boundary']['input_num'][-1] # current_boundary_input
        cbri = self.history_data['boundary']['r_i_dif'][-1] # current_boundary_release
        boundary_name_list = self.boundary_intersections['name_list']
        self.history_data['boundary']['max_release'].append((max(cbr), boundary_name_list[cbr.index(max(cbr))]))
        self.history_data['boundary']['max_input'].append((max(cbi), boundary_name_list[cbi.index(max(cbi))]))
        self.history_data['boundary']['max_r_i_dif'].append((max(cbri), boundary_name_list[cbri.index(max(cbri))]))
        self.history_data['boundary']['sum_release'].append(round(sum(cbr),4))
        self.history_data['boundary']['sum_input'].append(round(sum(cbi),4))
        self.history_data['boundary']['sum_r_i_dif'].append(round(sum(cbri),4))
        self.history_data['boundary']['min_release'].append((min(cbr), boundary_name_list[cbr.index(min(cbr))]))
        self.history_data['boundary']['min_input'].append((min(cbi), boundary_name_list[cbi.index(min(cbi))]))
        self.history_data['boundary']['min_r_i_dif'].append((min(cbri), boundary_name_list[cbri.index(min(cbri))]))
        print("sum_r_i_dif: {}".format(round(sum(cbri),4)))

        # print("max_release: {} / {}, sum_release: {}, min_release {} / {}".format(max(cbr), boundary_name_list[cbr.index(max(cbr))], round(np.sum(cbr),4), min(cbr), boundary_name_list[cbr.index(min(cbr))]))
        # print("max_input: {} / {}, sum_input: {}, min_input {} / {}".format(max(cbi), boundary_name_list[cbi.index(max(cbi))], round(np.sum(cbi),4), min(cbi), boundary_name_list[cbi.index(min(cbi))]))
        # print("max_r_i_dif: {} / {}, sum_r_i_dif: {}, min_r_i_dif {} / {}".format(max(cbri), boundary_name_list[cbri.index(max(cbri))], round(np.sum(cbri),4), min(cbri), boundary_name_list[cbri.index(min(cbri))]))
    
    def show_car_num(self):
        print("car_num inside: {}, ql_num inside: {}".format(len(self.history_data['veh_log']['inside_list']), self.history_data['car_num_inside']['waiting'][-1]))
    

    def process_state(self, state):
        current_states = []
        current_outputs = []
        for i in range(len(state)):
            # log statistic state
            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, statistic_incoming_state, mean_speed = self.get_state_detail_many_seg_all_lane(roads)
            arrive_left_times = self.env.list_intersection[i].dic_vehicle_arrive_leave_time
            for lane in location_all_direction_dict:
                statistic_state[lane]['stay_time'] = {}
                statistic_state[lane]['occupancy'] = len(statistic_state[lane]['veh2pos'])/(statistic_state[lane]['road_length']//self.car_spacing)
                for veh in statistic_state[lane]['veh2cell']:
                    enter_time = arrive_left_times[veh]["enter_time"]
                    current_time = self.env.current_time
                    statistic_state[lane]['stay_time'][veh] = current_time - enter_time

            current_states.append(statistic_state)
            current_outputs.append(statistic_incoming_state)
        return current_states, current_outputs

    def find_min_spacing(self, veh2pos):
        veh_pos_list = list(veh2pos.values())
        veh_pos_list.sort()
        min_spacing = float('inf')
        for i in range(len(veh_pos_list)):
            for j in range(i + 1, len(veh_pos_list)):
                spacing = abs(veh_pos_list[i]-veh_pos_list[j])
                if spacing < min_spacing:
                    min_spacing = spacing
        return min_spacing

    def get_state_detail_many_seg_all_lane(self, roads):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        divide the lane into many seg
        tag1
        """
        lane_queues = self.env.eng.get_lane_waiting_vehicle_count()
        lane_vehicles = self.env.eng.get_lane_vehicles()

        # init statistic info & get queue info
        statistic_state = {}
        statistic_state_incoming = {}
        outgoing_lane_speeds = []
        for r in roads:
            # road info
            location = roads[r]["location"]
            road_length = float(roads[r]["length"])

            # get queue info
            if roads[r]["type"] == "outgoing":  ## lane that target to this inter
                if roads[r]["go_straight"] is not None:
                    queue_len = 0.0
                    for lane in roads[r]["lanes"]["go_straight"]:
                        queue_len += lane_queues[f"{r}_{lane}"]
                    statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(self.seg_num)],
                                                                                        "ql_cells": [0 for _ in range(self.seg_num)],
                                                                                        "queue_len": queue_len,
                                                                                        "out_of_lane": 0, 
                                                                                        "veh2cell":{},
                                                                                        "avg_wait_time": 0.0,
                                                                                        "wait_time": {},
                                                                                        "veh2pos": {},
                                                                                        "road_length": road_length}

                if roads[r]["turn_left"] is not None:
                    queue_len = 0.0
                    for lane in roads[r]["lanes"]["turn_left"]:
                        queue_len += lane_queues[f"{r}_{lane}"]
                    statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(self.seg_num)],
                                                                                        "ql_cells": [0 for _ in range(self.seg_num)],
                                                                                        "queue_len": queue_len,
                                                                                        "out_of_lane": 0, 
                                                                                        "veh2cell":{},
                                                                                        "avg_wait_time": 0.0,
                                                                                        "wait_time": {},
                                                                                        "veh2pos": {},
                                                                                        "road_length": road_length}
                if roads[r]["turn_right"] is not None:
                    queue_len = 0.0
                    for lane in roads[r]["lanes"]["turn_right"]:
                        queue_len += lane_queues[f"{r}_{lane}"]
                    statistic_state[f"{location_dict_short[roads[r]['location']]}R"] = {"cells": [0 for _ in range(self.seg_num)],
                                                                                        "ql_cells": [0 for _ in range(self.seg_num)],
                                                                                        "queue_len": queue_len,
                                                                                        "out_of_lane": 0, 
                                                                                        "veh2cell":{},
                                                                                        "avg_wait_time": 0.0,
                                                                                        "wait_time": {},
                                                                                        "veh2pos": {},
                                                                                        "road_length": road_length}

                # get vehicle position info
                straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
                left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
                right_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_right"]]
                lanes = straight_lanes + left_lanes + right_lanes

                for lane in lanes:
                    waiting_times = []
                    vehicles = lane_vehicles[lane]

                    # collect lane group info
                    if location == "North":
                        if lane in straight_lanes:
                            lane_group = 0
                        elif lane in left_lanes:
                            lane_group = 1
                        elif lane in right_lanes:
                            lane_group = 2
                    elif location == "South":
                        if lane in straight_lanes:
                            lane_group = 3
                        elif lane in left_lanes:
                            lane_group = 4
                        elif lane in right_lanes:
                            lane_group = 5
                    elif location == "East":
                        if lane in straight_lanes:
                            lane_group = 6
                        elif lane in left_lanes:
                            lane_group = 7
                        elif lane in right_lanes:
                            lane_group = 8
                    elif location == "West":
                        if lane in straight_lanes:
                            lane_group = 9
                        elif lane in left_lanes:
                            lane_group = 10
                        elif lane in right_lanes:
                            lane_group = 11
                    else:
                        lane_group = -1

                    # collect lane cell info
                    for veh in vehicles:
                        veh_info = self.env.eng.get_vehicle_info(veh)
                        lane_pos = road_length - float(veh_info["distance"])
                        statistic_state[location_all_direction_dict[lane_group]]["veh2pos"][veh] = lane_pos
                        # update statistic state
                        seg_length = road_length/self.seg_num
                        gpt_lane_cell = int(lane_pos//seg_length)
                        statistic_state[location_all_direction_dict[lane_group]]["veh2cell"][veh] = gpt_lane_cell
                        veh_waiting_time = self.env.waiting_vehicle_list[veh]['time'] if veh in self.env.waiting_vehicle_list else 0.0
                        statistic_state[location_all_direction_dict[lane_group]]["wait_time"][veh] = veh_waiting_time
                        if veh in self.env.waiting_vehicle_list:
                            waiting_times.append(veh_waiting_time)
                        if gpt_lane_cell >= self.seg_num:
                            statistic_state[location_all_direction_dict[lane_group]]["out_of_lane"] += 1
                        else:
                            # speed > 0.1 m/s are approaching vehicles
                            speed = float(veh_info["speed"])
                            if speed > 0.1:
                                statistic_state[location_all_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1                        
                                outgoing_lane_speeds.append(speed)
                            else:
                                statistic_state[location_all_direction_dict[lane_group]]["ql_cells"][gpt_lane_cell] += 1           
                                
                    avg_wait_time = np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                    statistic_state[location_all_direction_dict[lane_group]]["avg_wait_time"] = avg_wait_time


            # incoming lanes
            else: ## lane that output from this inter
                queue_len = 0.0
                for lane in range(3):
                    queue_len += lane_queues[f"{r}_{lane}"]
                # change the incoming direction to output direction

                statistic_state_incoming[incoming2output[roads[r]['location']]] = {"cells": [0 for _ in range(self.seg_num)],
                                                                                    "ql_cells": [0 for _ in range(self.seg_num)],
                                                                                    "out_of_lane": 0,
                                                                                    "veh2cell":{},
                                                                                    "queue_len": queue_len}
                incoming_lanes = [f"{r}_{idx}" for idx in range(3)]

                for lane in incoming_lanes:
                    vehicles = lane_vehicles[lane]

                    # collect lane group info
                    if location == "North": # this location is incoming direction, we need to change it to output direction
                        lane_group = 1
                    elif location == "South":
                        lane_group = 0
                    elif location == "East":
                        lane_group = 3
                    elif location == "West":
                        lane_group = 2
                    else:
                        lane_group = -1

                    # collect lane cell info
                    for veh in vehicles:
                        veh_info = self.env.eng.get_vehicle_info(veh)
                        lane_pos = road_length - float(veh_info["distance"])

                        # update statistic state
                        seg_length = road_length/self.seg_num
                        gpt_lane_cell = int(lane_pos//seg_length)
                        statistic_state_incoming[location_incoming_dict[lane_group]]["veh2cell"][veh] = gpt_lane_cell
                        if gpt_lane_cell >= self.seg_num:
                            statistic_state_incoming[location_incoming_dict[lane_group]]["out_of_lane"] += 1
                            
                        else:
                            # speed > 0.1 m/s are approaching vehicles
                            if float(veh_info["speed"]) > 0.1:
                                statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1
                            else:
                                statistic_state_incoming[location_incoming_dict[lane_group]]["ql_cells"][gpt_lane_cell] += 1

        mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

        return statistic_state, statistic_state_incoming, mean_speed

    def update_history_data(self, states, next_states, current_outputs, next_outputs, action_list):
        self.history_data['release_car_num'].append(0)
        self.history_data['input_car_num'].append(0)
        # self.history_data['car_num'].append(0)
        self.history_data['car_num_out_of_lane'].append(0)
        self.history_data["avg_ql_cells"].append([])
        self.history_data['boundary']['release_num'].append([])
        self.history_data['boundary']['input_num'].append([])
        self.history_data['boundary']['r_i_dif'].append([])
        self.history_data["car_num_outside"]['waiting'].append(0)
        self.history_data["car_num_outside"]['running'].append(0)
        self.history_data["car_num_outside"]['total'].append(0)
        self.history_data["veh_num"].append(0)
        self.history_data["ql_num"].append(0)
        self.release_data = []
        # self.history_data['veh_log'] = {}
        # self.history_data['veh_log']['inside_veh'] = []
        # self.history_data['veh_log']['outside_veh'] = set()
        # self.history_data['veh_log']['all_veh'] = set()
        self.history_data['veh_log']['outside_list'] = []
        
        
        for i, s in enumerate(states):
            p = code2action(action_list[i])
            lane1 = p[:2]
            lane2 = p[2:]
            self.updata_history_data_release_range(lane1, s, next_states[i])
            self.updata_history_data_release_range(lane2, s, next_states[i])
            ql_cells = []
            self.release_data.append(0) 
            for lane in location_all_direction_dict:
                self.updata_history_data_per_lane(lane, s, next_states[i], ql_cells)

            self.history_data["avg_ql_cells"][-1].append(np.mean(ql_cells))
            if i in self.boundary_intersections['id_list']:
                release_num, input_num, car_num_out_side = self.count_boundary_data(i, states[i], next_states[i], current_outputs[i], next_outputs[i])
                self.history_data['boundary']['release_num'][-1].append(release_num)
                self.history_data['boundary']['input_num'][-1].append(input_num)
                self.history_data['boundary']['r_i_dif'][-1].append(release_num - input_num)
                self.history_data["car_num_outside"]['waiting'][-1] += car_num_out_side['waiting']
                self.history_data["car_num_outside"]['running'][-1] += car_num_out_side['running']
                self.history_data["car_num_outside"]['total'][-1] += car_num_out_side['total']
        # self.history_data['boundary']['r_i_dif'].append(list(np.array(self.history_data['boundary']['release_num'][-1]) - np.array(self.history_data['boundary']['input_num'][-1])))
        self.refine_history_data_release_range()
        for inter in self.env.list_intersection:
            self.history_data["veh_num"][-1] += sum(inter.dic_feature['lane_num_vehicle'])
            self.history_data["ql_num"][-1] += sum(inter.dic_feature['lane_num_waiting_vehicle_in'])
        self.history_data["car_num_inside"]['total'].append(self.history_data["veh_num"][-1] - self.history_data["car_num_outside"]['total'][-1])
        self.history_data["car_num_inside"]['waiting'].append(self.history_data["ql_num"][-1] - self.history_data["car_num_outside"]['waiting'][-1])
        self.history_data["car_num_inside"]['running'].append(self.history_data["car_num_inside"]['total'][-1] - self.history_data["car_num_inside"]['waiting'][-1])
        # self.history_data['veh_log']['all_veh/outside_veh'] = self.history_data['veh_log']['all_veh'] - self.history_data['veh_log']['outside_veh']
        self.history_data["car_num_inside"]['r_i'].append(len(self.history_data['veh_log']['inside_list']))

    def count_boundary_data(self, i, state, next_state, current_output, next_output):
        release_num = 0
        input_num = 0
        car_num_out_side = {}
        car_num_out_side['waiting'] = 0
        car_num_out_side['running'] = 0
        car_num_out_side['total'] = 0

        neighbor_list = self.get_neighbor_list(i)
        assert len(neighbor_list) < 4
        four_location = ['North', 'South', 'West', 'East']
        location_list = [neighbor['location'] for neighbor in neighbor_list]
        outside_location_list = [loc for loc in four_location if loc not in location_list]
        for loc in outside_location_list:
            current_output_veh_list = list(current_output[loc[0]]['veh2cell'].keys())
            next_output_veh_list = list(next_output[loc[0]]['veh2cell'].keys())
            release_vehs = [veh for veh in next_output_veh_list if veh not in current_output_veh_list]
            # release_num += len(release_vehs)
            for veh in release_vehs:
                if veh in self.history_data['veh_log']['inside_list']:
                    self.history_data['veh_log']['inside_list'].remove(veh)
                    release_num += 1
            # self.history_data['veh_log']['inside_veh'] = self.history_data['veh_log']['inside_veh'] - set(release_vehs)
            assert current_output[loc[0]]['queue_len'] == sum(current_output[loc[0]]['ql_cells'])

            # car_num_out_side['waiting'] += next_output[loc[0]]['queue_len'] 
            # car_num_out_side['running'] += sum(next_output[loc[0]]['cells'])
            # car_num_out_side['total'] += len(next_output_veh_list)
            
            current_input_veh_list = []
            next_input_veh_list = []
            
            for direc in ['L','T','R']:
                current_input_veh_list.extend(list(state[loc[0] + direc]['veh2cell'].keys()))
                next_input_veh_list.extend(list(next_state[loc[0] + direc]['veh2cell'].keys()))

                car_num_out_side['waiting'] += next_state[loc[0] + direc]['queue_len']
                car_num_out_side['running'] += sum(next_state[loc[0] + direc]['cells'])

            car_num_out_side['total'] += len(next_input_veh_list)

            input_vehs = [veh for veh in current_input_veh_list if veh not in next_input_veh_list]
            input_num += len(input_vehs)

            # self.history_data['veh_log']['inside_veh'].update(set(input_vehs))
            self.history_data['veh_log']['inside_list'].extend(set(input_vehs))
            # self.history_data['veh_log']['outside_veh'].update(set(next_output_veh_list + next_input_veh_list))
            self.history_data['veh_log']['outside_list'].extend(next_output_veh_list + next_input_veh_list)

        return release_num, input_num, car_num_out_side

    def updata_history_data_per_lane(self, lane, state, next_state, ql_cells):
        lane_vehs = state[lane]["veh2cell"]
        lane_vehs_next = next_state[lane]["veh2cell"]
        lane_vehs_list = list(lane_vehs.keys())
        lane_vehs_keys_next = list(lane_vehs_next.keys())
        depart_vehs = []
        stay_vehs = []
        for veh in lane_vehs_list:
            if veh in lane_vehs_keys_next:
                stay_vehs.append(veh)
            else:
                depart_vehs.append(veh)

        self.history_data['car_num_out_of_lane'][-1] += next_state[lane]["out_of_lane"]
        self.history_data['release_car_num'][-1] += len(depart_vehs)
        self.release_data[-1] += len(depart_vehs)
        self.history_data['input_car_num'][-1] += len(lane_vehs_keys_next) - len(stay_vehs)
        ql_cells.append(np.count_nonzero(next_state[lane]['ql_cells']))

    def refine_history_data_release_range(self):
        for lane in self.history_data["release_range"]:
            self.history_data["release_range"][lane] = fix_decreasing_list(self.history_data["release_range"][lane])
            
    def updata_history_data_release_range(self, lane, state, next_state):
        ql_num = state[lane]['queue_len']
        end_cell = self.identify_last_cell(state[lane]["veh2cell"], next_state[lane]["veh2cell"])
        


        if end_cell:
            if ql_num in self.history_data["release_range"][lane]:
                if end_cell > self.history_data["release_range"][lane][ql_num]:
                    self.history_data["release_range"][lane][ql_num] = end_cell
            else:
                self.history_data["release_range"][lane][ql_num] = end_cell
            if state[lane]['queue_len'] not in self.history_data['effective_release_log']:
                self.history_data['effective_release_log'][state[lane]['queue_len']] = [state[lane], end_cell]
            elif end_cell > self.history_data['effective_release_log'][state[lane]['queue_len']][1]:
                self.history_data['effective_release_log'][state[lane]['queue_len']] = [state[lane], end_cell]

    def identify_last_cell(self, lane_vehs, lane_vehs_next):
        lane_vehs_list = list(lane_vehs.keys())
        lane_vehs_keys_next = list(lane_vehs_next.keys())
        depart_vehs = []
        stay_vehs = []
        for veh in lane_vehs_list:
            if veh in lane_vehs_keys_next:
                stay_vehs.append(veh)
            else:
                try:
                    vehicle_info = self.env.eng.get_vehicle_info(veh)
                    if vehicle_info:
                        depart_vehs.append(veh)
                except Exception:
                    pass
        if stay_vehs:
            min_veh_stay = min(stay_vehs, key=lambda x: lane_vehs_next[x])
            upper_bound = lane_vehs[min_veh_stay] - lane_vehs_next[min_veh_stay]
        else:
            upper_bound = None

        if depart_vehs:
            max_veh_depart = max(depart_vehs, key=lambda x: lane_vehs[x])
            lower_bound = lane_vehs[max_veh_depart]
        else:
            lower_bound = None
        last_cell = lower_bound

        if upper_bound:
            if not last_cell:
                last_cell = upper_bound - 1
            elif last_cell > upper_bound:
                last_cell = upper_bound - 1
        
        return last_cell

    def estimate_effective_range(self, state):
        # self.car_speed = 11
        # self.
        range_list = []
        for lane in location_direction_dict:
            ql_num = state[lane]['queue_len']
            exit_qls = list(self.history_data["release_range"][lane].keys())
            if ql_num in self.history_data["release_range"][lane]:
                range_list.append(self.history_data["release_range"][lane][ql_num])
            elif len(exit_qls):
                # exit_qls = list(self.history_data["release_range"][lane].keys())
                exit_qls = np.array(exit_qls)
                closest_ql = exit_qls[np.argmin(np.abs(exit_qls - ql_num))]
                if np.abs(closest_ql) < 5 or ql_num < closest_ql:
                    range_list.append(self.history_data["release_range"][lane][closest_ql])
                elif self.history_data["release_range"][lane][closest_ql] < self.default_release_range:
                    range_list.append(self.history_data["release_range"][lane][closest_ql])
                else:
                    range_list.append(self.default_release_range)
            else:
                range_list.append(self.default_release_range)
        return range_list

    def estimate_effective_range_new(self, state):
        car_speed = 11.11
        effective_range_distance = car_speed * self.signal_time
        range_list = []
        for lane in location_direction_dict:
            road_length = state[lane]['road_length']
            seg_length = road_length/self.seg_num
            effective_range_cell = int(effective_range_distance//seg_length)
            if effective_range_cell >= self.seg_num:
                effective_range_cell = self.seg_num - 1
            range_list.append(effective_range_cell)
        return range_list

    def get_neighbor_list(self, inter_id):
        n_list = []
        inter_name = self.env.list_intersection[inter_id].inter_name
        inter_list = list(self.env.intersection_dict.keys())
        intersection = self.env.intersection_dict[inter_name]
        roads = deepcopy(intersection["roads"])

        neighbor_list = [inter for inter in self.env.traffic_light_node_dict[inter_name]['neighbor_ENWS'] if inter] #inter_name

        road_list = list(roads.keys())
        road2inter = [r.replace("road", "intersection")[:-2] for r in road_list] 
        
        neighbor2loc = {inter: roads[road_list[i]]['location'] for i, inter in enumerate(road2inter) if inter in neighbor_list}
        for neighbor_inter_name in neighbor_list:
            n_list.append({"id":inter_list.index(neighbor_inter_name), "name":neighbor_inter_name,  "location":neighbor2loc[neighbor_inter_name]})
        return n_list

    def train_test(self):
        from utils.local_tracking import wandb
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )
        self.test(logger, 0)

    def get_norm_reward(self, state):
        rewards = []

        for i in range(len(state)):
            vehicle_num = 0
            queue_length = 0

            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]['queue_len']

                vehicle_num += statistic_state[lane]['queue_len']
                for cell in range(len(statistic_state[lane]['cells'])):
                    vehicle_num += statistic_state[lane]['cells'][cell]

            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(reward)

        return rewards

