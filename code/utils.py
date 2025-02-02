import numpy as np
import torch.utils.data as data
import os
import yaml
from easydict import EasyDict as edict
from scipy.signal import wiener

def read_config(config_path):
    """Считывает конфигурацию из YAML файла и преобразует ее в easydict.

    Args:
        config_path (str): Путь к YAML файлу.

    Returns:
        easydict: Объект easydict с параметрами конфигурации.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) # используйте safe_load для безопасности
    return edict(config_dict)

def wiener_filter(noisy_image, frame):
    filtered_image = wiener(noisy_image, frame)
    return filtered_image
