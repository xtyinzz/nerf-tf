import os
import numpy as np
import imageio
# import torch
import h5py
from run_nerf_helpers import *
# from .data import train_val_indices

def train_val_indices(dataset_size, valid_split):
  random_seed= 42
  # Creating data indices for training and validation splits:
  indices = np.arange(dataset_size)
  np.random.seed(random_seed)
  np.random.shuffle(indices)
  val_count = int(np.floor(valid_split * dataset_size))
  train_indices, val_indices = indices[val_count:], indices[:val_count]

  return train_indices, val_indices


def load_nyx_data(fdir, testSplit=0.1):
  nyx = {}
  with h5py.File(os.path.join(fdir,"nyx.hdf5"), "r") as f:
    for k,v in f.items():
      nyx[k] = np.array(v)
      
  # renderPoses = {}
  # with h5py.File(os.path.join(fdir,"poses.hdf5"), "r") as f:
  #   for k,v in f.items():
  #     print(k,v)
  #     renderPoses[k] = np.array(v)
      
  # p = nyx['p']
  # viewDir = nyx['dir']
  # camPos = nyx['cam']
  mv = nyx['mv']
  c2w = nyx['mvInvs']
  nearFar = nyx['clip']
  bbox = nyx['bbox']
  
  imgs = []
  for i in range(len(mv)):
    img = imageio.imread(os.path.join(fdir, "img", f"{i:03}.png"))
    imgs.append(img)
  imgs = np.concatenate([imgs], 0)
  imgs = (imgs / 255.).astype(np.float32)
  hw = imgs.shape[1:3]
  iTrain, iVal = train_val_indices(len(mv), testSplit)
  
  iSplit = [iTrain, iVal, []]
  
  return imgs, c2w, hw, nearFar, bbox, iSplit, None