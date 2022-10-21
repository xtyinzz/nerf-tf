import sklearn
import torch
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler, Subset
from util.utils import cpuStats
from os import path
from tqdm import tqdm, trange
from util.utils import report_gpumem, cpuStats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import vtk
from vtkmodules.util import numpy_support


def parse_filenames(pfiles):
  '''
  Given path to a txt file containing data file paths, whitespace seperated,
  Return file names as a list.
  '''
  with open(pfiles, 'r') as f:
    files = [line.rstrip() for line in f]
  return files

def write_stats(particle, outpath):
  '''
  get min max value of particle data (N, C) and write to npy file

  return: (S, C), S={min, max}
  '''
  attrmin = particle.min(0, keepdims=True)
  attrmax = particle.max(0, keepdims=True)
  stats = np.concatenate([attrmin, attrmax], 0)
  np.save(outpath, stats)
  print(f'stats shaped {stats.shape} saved {outpath}')


def minmax_scale(x, new_min=-1., new_max=1., x_min=None, x_max=None):
  if x_min is None:
    x_min = x.min(1, keepdim=True)[0]
  if x_max is None:
    x_max = x.max(1, keepdim=True)[0]
  return (x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min

def standardization(x:torch.Tensor, dim=None):
  if dim is None:
    xmean = x.mean()
    xstd = x.std()
  else:
    xmean = torch.mean(x, dim=dim, keepdim=True)[0]
    xstd = torch.min(x, dim=dim, keepdim=True)[0]
  return (x - xmean / xstd)

def normalization(x:np.ndarray, new_min=-1, new_max=1, dim=None):
  if dim is None:
    curr_max = x.max()
    curr_min = x.min()
  else:
    curr_max = np.max(x, dim, keepdims=True)
    curr_min = np.min(x, dim, keepdims=True)
  return (x - curr_min) / (curr_max - curr_min) * (new_max - new_min) + new_min

def train_val_indices(dataset_size, valid_split):
  random_seed= 42
  # Creating data indices for training and validation splits:
  indices = np.arange(dataset_size)
  np.random.seed(random_seed)
  np.random.shuffle(indices)
  val_count = int(np.floor(valid_split * dataset_size))
  train_indices, val_indices = indices[val_count:], indices[:val_count]

  return train_indices, val_indices

def train_val(dataset: Dataset, valid_split):
  validation_split = .2
  shuffle_dataset = True
  random_seed= 42

  # Creating data indices for training and validation splits:
  dataset_size = len(dataset)
  train_indices, val_indices = train_val_indices(dataset_size, valid_split)

  # Creating training sampler and validation dataset:
  train_sampler = SubsetRandomSampler(train_indices)
  val_dataset = Subset(dataset, val_indices)
  
  return train_sampler, val_dataset

  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
  validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=valid_sampler)
  validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# class TransPipeline():
#   def __init__(self):
#     self.norm = NormScaler()
#     self.stand = StandScaler()

def fitTransPipeline(x: np.ndarray):
  pp = Pipeline1D([
    ('stand', StandardScaler()),
    ('norm', MinMaxScaler((0, 1))),
  ])
  x_shape = x.shape
  x = pp.fit_transform(x)
  return x, pp

# global pipeline (norm and stand coordinates in SphericalDataset's case)
class Pipeline1D(Pipeline):
  def fit_transform(self, X, y=None, **fit_params):
    x_shape = X.shape
    X = X.reshape(-1, 1)
    X = super().fit_transform(X, y, **fit_params)
    return X.reshape(x_shape)

  def fit(self, X, y=None, **fit_params):
    x_shape = X.shape
    X = X.reshape(-1, 1)
    return super()._fit(X, y, **fit_params)
  
  def transform(self, X):
    x_shape = X.shape
    X = X.reshape(-1, 1)
    X = super()._transform(X)
    return X.reshape(x_shape)

  def inverse_transform(self, X):
    x_shape = X.shape
    X = X.reshape(-1, 1)
    X = super()._inverse_transform(X)
    return X.reshape(x_shape)

'''
    data_dir: "../data/",
    curv_idx: [0,1,2],
    cart_idx: [3,4,5],
'''
class SphericalDataset(Dataset):
  def __init__(self, data_path, curv_idx, cart_idx, intrans=fitTransPipeline, outtrans=fitTransPipeline):
    self.data_path = data_path
    self.coords = np.load(data_path)
    self.dims = self.coords.shape[:-1]
    self.curv_idx = curv_idx
    self.cart_idx = cart_idx
    self.curv = self.coords[..., curv_idx]
    self.cart = self.coords[..., cart_idx]
    assert len(self.curv) == len(self.cart)

    if intrans is not None:
      print("transforming inputs")
      self.cart_prep, self.inpp = fitTransPipeline(self.cart.reshape(-1, len(cart_idx)))
      # print(self.cart.mean())
    if outtrans is not None:
      print("transforming outputs")
      self.curv_prep, self.outpp = fitTransPipeline(self.curv.reshape(-1, len(curv_idx)))
      # print(self.curv.mean())
  
    self.cart_prep = torch.tensor(self.cart_prep)
    self.curv_prep = torch.tensor(self.curv_prep)

  def __len__(self):
    return len(self.curv_prep)

  def __getitem__(self, idx):
    return self.cart_prep[idx], self.curv_prep[idx]