# TensorFlow 2.8 generated protobuf descriptors require the Python implementation
# with the existing newer protobuf package. Set before importing the TF agents.
import os
import sys
from pathlib import Path
_tf_compat=Path(__file__).resolve().parents[1]/'.venv/compat/tensorflow28'
if (_tf_compat/'google/protobuf').is_dir():
    loaded=sys.modules.get('google.protobuf')
    if loaded is not None and getattr(loaded,'__version__','')!='3.20.3':
        raise RuntimeError('Run legacy RL in a fresh process so its isolated protobuf can load before TensorFlow')
    sys.path.insert(0,str(_tf_compat))
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
from models.random_agent import RandomAgent
from models.fixedtime_agent import FixedtimeAgent
from models.maxpressure_agent import MaxPressureAgent
from models.efficient_maxpressure_agent import EfficientMaxPressureAgent
from models.mplight_agent import MPLightAgent
from models.colight_agent import CoLightAgent
from models.presslight_one import PressLightAgentOne
from models.advanced_mplight_agent import AdvancedMPLightAgent
from models.advanced_maxpressure_agent import AdvancedMaxPressureAgent
from models.simple_dqn_one import SimpleDQNAgentOne
from models.attendlight_agent import AttendLightAgent
from models.chatgpt import (ChatGPTTLCS_Wait_Time_Forecast, ChatGPTTLCS_Commonsense)

DIC_AGENTS = {
    "Random": RandomAgent,
    "Fixedtime": FixedtimeAgent,
    "MaxPressure": MaxPressureAgent,
    "EfficientMaxPressure": EfficientMaxPressureAgent,
    "AdvancedMaxPressure": AdvancedMaxPressureAgent,

    "EfficientPressLight": PressLightAgentOne,
    "EfficientColight": CoLightAgent,
    "EfficientMPLight": MPLightAgent,
    "MPLight": MPLightAgent,
    "Colight": CoLightAgent,

    "AdvancedMPLight": AdvancedMPLightAgent,
    "AdvancedColight": CoLightAgent,
    "AdvancedDQN": SimpleDQNAgentOne,
    "Attend": AttendLightAgent,
    "ChatGPTTLCSWaitTimeForecast": ChatGPTTLCS_Wait_Time_Forecast,
    "ChatGPTTLCSCommonsense": ChatGPTTLCS_Commonsense
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

from .env_defaults import dic_traffic_env_conf

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_CHATGPT_AGENT_CONF = {
    "GPT_VERSION": "gpt-4",
    "LOG_DIR": "../GPT_logs"
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}