import sys
from importlib import import_module
import os


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
    return cfg_dict

if __name__ == "__main__":
    aa = parse_py_config("configs/single_line_chinese.py")
    print(aa)
    print(aa.ee)