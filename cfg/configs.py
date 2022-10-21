import os
import torch
import yaml
from data.data import VBHGridDataset, SphericalDataset, SphericalBlockDataset, Phys2CompDataset
from network.network import MLP, AttentionNetwork, ResMLP, Siren
# parse config files; reuse for your own project
class Config():
  '''
  extendable abstraction of parsing YAML config file including following format where:
    - strings: keys
    - ...: user-speified
    - []: optional or example keys

  # keys that didn't provide abstractions/methods:
  # 1.
  # logs related varialbes fo your choice
  # (e.g. dir of checkpoint, metrics, stats ...)
  logs: {
    ...
  }
  ----------
  Attributes
    - config: dict of parsed yaml, can be used directly w/o abstraction if desired

  '''
  def __init__(self, cfg_path) -> None:
    with open(f'cfg/{cfg_path}', 'r') as f:
      config = yaml.load(f, yaml.FullLoader)
    self.config = config

  def get_dataset_param(self):
    '''
    dataset: {
      [type: ...]
      param: { ... } # parameter for your pytorch Dataset object
    }
    if type not specified, return the dataset dictionary
    '''
    cfg = self.config['dataset']
    param = cfg['param']
    return param
  
  def get_dataset(self):
    cfg = self.config['dataset']
    name = cfg['type']
    param = cfg['param']
    dataset = None
    if name == 'mantle_coord':
      dataset = SphericalDataset(**param)
    elif name == "mantle_coord_block":
      dataset = SphericalBlockDataset(**param)
    elif name == "vls_2022":
      dataset = Phys2CompDataset(**param)
    return dataset
  
  def get_optim(self, model):
    '''
    get nn.optim: supporting "Adam", "AdamW"

    optim: {
      type: ..., # name of supported torch.optim (e.g. Adam, AdamW, etc.)
      param: { ... } # parameter for your torch.optim object
    }
    '''
    # dafault optimizer: Adam, lr=0.001, betas=(0.9, 0.999) if not provided config
    optim = torch.optim.Adam(model.parameters())
    if 'optim' in self.config:
      cfg = self.config['optim']
      name = cfg['type']
      param = cfg['param']
      if name == 'AdamW':
        optim = torch.optim.AdamW(model.parameters(), **param)
      elif name == 'Adam':
        optim = torch.optim.Adam(model.parameters(), **param)
    return optim
  
  def get_model_param(self):
    '''
    model: {
      type: ..., # name of your network models. MODIFY get_model() to parse your object
      param: { ... } # parameter for your torch.nn.Module object
    }
    '''
    cfg = self.config['model']
    name = cfg['type']
    param = cfg['param']
    return param

  def get_model(self):
    '''
    model: {
      type: ..., # name of your network models. MODIFY get_model() to parse your object
      param: { ... } # parameter for your torch.nn.Module object
    }
    '''
    cfg = self.config['model']
    name = cfg['type']
    param = cfg['param']
    model = None
    if 'siren' == name:
      model = Siren(**param)
    elif 'mlp_pe' == name:
      model = MLP(**param)
    elif 'mlp_pe_res' == name:
      model = ResMLP(**param)
    elif 'attn_net' == name:
      model = AttentionNetwork(**param)
    return model

  def get_train_args(self):
    '''
    # training related variables of your choice
    # (e.g. maximum epoch, batch size, steps to log ckpt and/or metrics)
    train: {
      ...
      [
        max_epoch: ...,
        batch: ...,
        log_step: ...,
      ]
    }
    '''
    cfg = self.config['train']

    # fill in default values if not included in config
    if 'max_epoch' not in cfg:
      cfg['max_epoch'] = 100
    if 'batch' not in cfg:
      cfg['batch'] = 1
    if 'log_step' not in cfg:
      cfg['log_step'] = 1
    return cfg