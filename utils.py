import yaml
from easydict import EasyDict as edict
from tensorflow.python.ops import data_flow_ops
import tensorflow as tf

def load_yml(path):
    with open(path, 'r') as f:
        try:
            config  = yaml.load(f)
            print(config)
            return edict(config)
        except yaml.YAMLError as exc:
            print(exc)
	
