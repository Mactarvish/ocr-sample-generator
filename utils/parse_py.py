import sys
from importlib import import_module
import os
from omegaconf import OmegaConf


def parse_py_config(src_py_path):
    src_py_dir = os.path.dirname(src_py_path)
    sys.path.insert(0, src_py_dir)
    mod = import_module(os.path.splitext(os.path.basename(src_py_path))[0])
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }
    config = OmegaConf.create(cfg_dict, flags={"allow_objects": True}) # 套一个OmegaConf可以把每个键变成attribute
    return config

if __name__ == "__main__":
    aa = parse_py_config("pgs/pg.py")
    print(aa)
    print(aa.ee)