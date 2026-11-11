"""Shared, stage-independent configuration. No model loading or API calls here."""
import argparse
import hashlib
import json
import os
import math
from pathlib import Path

from dotenv import dotenv_values

# This pipeline is PyTorch-only. Do not import the legacy TensorFlow baseline
# stack (and its unrelated protobuf dependency) through Transformers utilities.
os.environ.setdefault('USE_TF', '0')

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'RUN_NAME': 'qwen15b_smoke', 'RANDOM_SEED': 42, 'OUTPUT_ROOT': './outputs',
    'TEACHER_MODEL': 'qwen3.7-plus',
    'TEACHER_BASE_URL': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'TEACHER_API_KEY_ENV': 'DASHSCOPE_API_KEY', 'TEACHER_TEMPERATURE': 0.1,
    'TEACHER_ENABLE_THINKING': False, 'TEACHER_MAX_TOKENS': 2048,
    'TEACHER_USE_ENV_PROXY': False,
    'TEACHER_TIMEOUT_SECONDS': 120, 'TEACHER_MAX_ATTEMPTS': 2,
    'TEACHER_MAX_SAMPLES': 5, 'TEACHER_RANDOM_SEED': 42, 'TEACHER_RESUME': True,
    'TRAFFIC_SAMPLE_PATH': './data/FinetuneData/SynTrain_sample_all.json',
    'TEACHER_OUTPUT_PATH': './data/FinetuneData/syn_rt_data.json',
    'STUDENT_MODEL_PATH': './llm_models/Qwen2.5-1.5B-Instruct', 'STUDENT_DTYPE': 'auto',
    'STUDENT_MAX_LENGTH': 8192, 'STUDENT_MAX_NEW_TOKENS': 1024,
    'DATA_EVAL_RATIO': 0.2, 'SFT_REQUIRE_RA_MATCH': True,
    'SFT_LORA_RANK': 8, 'SFT_LORA_ALPHA': 16, 'SFT_LORA_DROPOUT': 0.0,
    'SFT_LORA_TARGET': 'all-linear', 'SFT_LEARNING_RATE': 1e-4,
    'SFT_NUM_EPOCHS': 1, 'SFT_PER_DEVICE_BATCH_SIZE': 1,
    'SFT_GRADIENT_ACCUMULATION_STEPS': 1, 'SFT_GRADIENT_CHECKPOINTING': True,
    'SFT_QUANTIZATION_BITS': 0, 'SFT_MAX_STEPS': 2, 'SFT_LOGGING_STEPS': 1,
    'MERGE_DEVICE': 'cpu', 'PPO_LORA_RANK': 8, 'PPO_LORA_ALPHA': 16,
    'PPO_LORA_DROPOUT': 0.0, 'PPO_LEARNING_RATE': 1e-5,
    'PPO_NUM_EPOCHS': 1, 'PPO_BATCH_SIZE': 2, 'PPO_MINI_BATCH_SIZE': 1,
    'PPO_GRADIENT_ACCUMULATION_STEPS': 1, 'PPO_OPTIMIZATION_EPOCHS': 1,
    'PPO_MAX_UPDATES': 1, 'PPO_MAX_NEW_TOKENS': 1024, 'PPO_TEMPERATURE': 0.1,
    'PPO_GRADIENT_CHECKPOINTING': True, 'PPO_QUANTIZATION_BITS': 0,
}

DEFAULTS.update(PPO_LORA_TARGET='all-linear', EXPORT_SOURCE='ppo_best', EXPORT_OUTPUT_DIR='',
                EXPORT_DEVICE='cpu', EXPORT_DTYPE='auto', EXPORT_VALIDATE=True, EXPORT_AUTO=True,
                CITYFLOW_MODEL_PATH='', CITYFLOW_BACKEND='transformers', CITYFLOW_DATASET='synthetic_4x4',
                CITYFLOW_TRAFFIC_FILE='anon_4_4_synthetic_8000.json', CITYFLOW_RUN_COUNTS=60,
                CITYFLOW_MAX_NEW_TOKENS=1024, CITYFLOW_SEED=42)
YAML_FIELDS = {
    'experiment': {'run_name':'RUN_NAME', 'random_seed':'RANDOM_SEED', 'output_root':'OUTPUT_ROOT'},
    'teacher_data': {'source_path':'TRAFFIC_SAMPLE_PATH', 'output_path':'TEACHER_OUTPUT_PATH',
                     'max_samples':'TEACHER_MAX_SAMPLES', 'random_seed':'TEACHER_RANDOM_SEED', 'resume':'TEACHER_RESUME'},
    'data': {'eval_ratio':'DATA_EVAL_RATIO', 'sft_require_ra_match':'SFT_REQUIRE_RA_MATCH'},
    'export': {'source':'EXPORT_SOURCE','output_dir':'EXPORT_OUTPUT_DIR','device':'EXPORT_DEVICE',
               'dtype':'EXPORT_DTYPE','validate':'EXPORT_VALIDATE','auto_export':'EXPORT_AUTO'},
    'cityflow_test': {'model_path':'CITYFLOW_MODEL_PATH','backend':'CITYFLOW_BACKEND',
                      'dataset':'CITYFLOW_DATASET','traffic_file':'CITYFLOW_TRAFFIC_FILE',
                      'run_counts':'CITYFLOW_RUN_COUNTS','max_new_tokens':'CITYFLOW_MAX_NEW_TOKENS','seed':'CITYFLOW_SEED'},
}
for _stage in ('sft', 'ppo'):
    _p = _stage.upper() + '_'
    YAML_FIELDS[_stage] = {
        'lora': {'rank':_p+'LORA_RANK','alpha':_p+'LORA_ALPHA','dropout':_p+'LORA_DROPOUT',
                 'target_modules':_p+'LORA_TARGET'},
        **{k:_p+v for k,v in {'learning_rate':'LEARNING_RATE','num_epochs':'NUM_EPOCHS',
            'gradient_accumulation_steps':'GRADIENT_ACCUMULATION_STEPS',
            'gradient_checkpointing':'GRADIENT_CHECKPOINTING','quantization_bits':'QUANTIZATION_BITS'}.items()},
        **({k:_p+v for k,v in {'per_device_batch_size':'PER_DEVICE_BATCH_SIZE',
             'max_steps':'MAX_STEPS','logging_steps':'LOGGING_STEPS'}.items()} if _stage == 'sft'
           else {k:_p+v for k,v in {'batch_size':'BATCH_SIZE','mini_batch_size':'MINI_BATCH_SIZE',
             'optimization_epochs':'OPTIMIZATION_EPOCHS','max_updates':'MAX_UPDATES',
             'max_new_tokens':'MAX_NEW_TOKENS','temperature':'TEMPERATURE'}.items()})}
MODEL_KEYS = {k for k in DEFAULTS if k.startswith(('TEACHER_', 'STUDENT_'))} - {
    'TEACHER_MAX_SAMPLES', 'TEACHER_RANDOM_SEED', 'TEACHER_RESUME', 'TEACHER_OUTPUT_PATH'}


def read_yaml_config(path):
    import yaml
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f'Duplicate YAML key: {key}')
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    document = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    flat = {}
    def walk(value, schema, prefix=''):
        if not isinstance(value, dict):
            raise ValueError(f'{prefix or "config"} must be a mapping')
        for name, item in value.items():
            if name not in schema:
                raise ValueError(f'Unknown YAML field: {prefix}{name}')
            target = schema[name]
            if isinstance(target, dict):
                walk(item, target, prefix + str(name) + '.')
            else:
                if item is None and target in ('EXPORT_OUTPUT_DIR','CITYFLOW_MODEL_PATH'):
                    item = ''
                default = DEFAULTS[target]
                if isinstance(default, bool) and not isinstance(item, bool):
                    raise ValueError(f'{prefix}{name} must be a YAML boolean')
                if isinstance(default, int) and not isinstance(default, bool) and (isinstance(item, bool) or not isinstance(item, int)):
                    raise ValueError(f'{prefix}{name} must be an integer')
                if isinstance(default, float) and (isinstance(item, bool) or not isinstance(item, (int, float))):
                    raise ValueError(f'{prefix}{name} must be numeric')
                if isinstance(default, str) and not isinstance(item, str):
                    raise ValueError(f'{prefix}{name} must be a string')
                flat[target] = item
    walk(document, YAML_FIELDS)
    return flat


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    def scalar(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        raise TypeError(f'Cannot serialize {type(obj).__name__}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=scalar))
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class TrainingConfig:
    def __init__(self, env_file=None, overrides=None, config_file=None):
        env_file = Path(env_file) if env_file else ROOT / '.env'
        if not env_file.is_absolute():
            env_file = ROOT / env_file
        self.config_file = Path(config_file) if config_file else ROOT / 'config/config.yaml'
        if not self.config_file.is_absolute():
            self.config_file = ROOT / self.config_file
        secrets_and_models = dict(dotenv_values(env_file)) if env_file.exists() else {}
        secrets_and_models.update(os.environ)
        self.raw = secrets_and_models
        effective = {k:v for k,v in secrets_and_models.items() if k in MODEL_KEYS}
        effective.update(read_yaml_config(self.config_file))
        effective.update({k:v for k,v in (overrides or {}).items() if v is not None})
        self.values = {}
        for key, default in DEFAULTS.items():
            value = effective.get(key, default)
            if isinstance(default, bool):
                if str(value).lower() not in ('true', 'false', '1', '0'):
                    raise ValueError(f'{key} must be true/false')
                value = str(value).lower() in ('true', '1')
            else:
                value = type(default)(value)
            self.values[key] = value
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
        self.values['MERGE_DEVICE'] = self.values['EXPORT_DEVICE']
        if not 0 < self['DATA_EVAL_RATIO'] < 1:
            raise ValueError('DATA_EVAL_RATIO must be between 0 and 1')
        if not self['RUN_NAME'] or self['RUN_NAME'] in ('.','..') or Path(self['RUN_NAME']).name != self['RUN_NAME']:
            raise ValueError('RUN_NAME must be a directory name')
        for key in ('TEACHER_MAX_SAMPLES', 'TEACHER_MAX_TOKENS', 'TEACHER_MAX_ATTEMPTS',
                    'TEACHER_TIMEOUT_SECONDS', 'STUDENT_MAX_LENGTH', 'STUDENT_MAX_NEW_TOKENS',
                    'SFT_LORA_RANK', 'SFT_LORA_ALPHA', 'SFT_NUM_EPOCHS',
                    'SFT_PER_DEVICE_BATCH_SIZE', 'SFT_GRADIENT_ACCUMULATION_STEPS',
                    'PPO_LORA_RANK', 'PPO_LORA_ALPHA', 'PPO_NUM_EPOCHS', 'PPO_BATCH_SIZE',
                    'PPO_MINI_BATCH_SIZE', 'PPO_GRADIENT_ACCUMULATION_STEPS',
                    'PPO_OPTIMIZATION_EPOCHS', 'PPO_MAX_NEW_TOKENS'):
            if self[key] <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('SFT_LEARNING_RATE', 'PPO_LEARNING_RATE', 'TEACHER_TEMPERATURE', 'PPO_TEMPERATURE'):
            if self[key] <= 0:
                raise ValueError(f'{key} must be positive')
        for key in ('SFT_LORA_DROPOUT', 'PPO_LORA_DROPOUT'):
            if not 0 <= self[key] < 1:
                raise ValueError(f'{key} must be in [0, 1)')

    def __getitem__(self, key):
        return self.values[key]

    def path(self, key):
        p = Path(self[key]).expanduser()
        return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

    @property
    def out(self):
        return self.path('OUTPUT_ROOT') / self['RUN_NAME']

    @property
    def export_dir(self):
        return self.path('EXPORT_OUTPUT_DIR') if self['EXPORT_OUTPUT_DIR'] else self.out / 'trained_best_model'

    @property
    def cityflow_model(self):
        return self.path('CITYFLOW_MODEL_PATH') if self['CITYFLOW_MODEL_PATH'] else self.export_dir

    def secret(self):
        name = self['TEACHER_API_KEY_ENV']
        value = self.raw.get(name)
        if not value:
            raise ValueError(f'Missing teacher API key environment variable: {name}')
        return value

    def snapshot(self):
        # Only known nonsecret keys are saved. Never serialize self.raw.
        atomic_json(self.out / 'resolved_config.json', self.values)


def config_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--env-file', default=None)
    parser.add_argument('--config', dest='config_file', default=None)
    return parser
