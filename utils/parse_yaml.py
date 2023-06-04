import os
import yaml
from omegaconf import OmegaConf
from copy import deepcopy


# 如果yaml中的backbone字段是"backbone"，那么改成"backbone_cfg"，neck等key类似，这么一来后续的各种模型类型构造时的参数名称就更加合理了
def add_cfg_postfix(config):
    TO_ADD_CFG_KEYS = ["model", "backbone", "neck", "bbox_head", 
                       'anchor_generator', 'bbox_assigner', 'bbox_sampler', 'bbox_coder', 
                       'loss_cls', 'loss_conf', 'loss_xy', 'loss_wh', "pipeline"]
    if not isinstance(config, dict):
        return
    for key in list(config.keys()):
        if (isinstance(config[key], dict)):
            add_cfg_postfix(config[key])
        if key in TO_ADD_CFG_KEYS:
            new_key = key + "_cfg"
            config[new_key] = deepcopy(config[key])
    for key in list(config.keys()):
        if key in TO_ADD_CFG_KEYS:
            config.pop(key)

# 只解析yaml，不解析傻逼py
def parse_yaml_config(config_path):
    assert os.path.exists(config_path), config_path
    assert os.path.splitext(config_path)[-1] == ".yaml", "你脑子里有泡啊，用锤子py，老老实实地用yaml"
    config = yaml.unsafe_load(open(config_path))
    print("执行yaml字段后缀补加，后边发现了没见过的字段不要惊讶哦")
    add_cfg_postfix(config)
    config = OmegaConf.create(config, flags={"allow_objects": True}) # 套一个OmegaConf可以把每个键变成attribute

    # TODO
    # check_validation(config)
    return config
