import numpy as np
import re
import os

# read .vec file with 3 integers of dimensions and floats for the rest:
# as numpy array of (xdim, ydim, zdim, 3)
#   xdim ydim zdim
#   vx vy vz
#   vx vy vz
#   ...

def read_vec(fname):
  c_intsize = 4
  with open(fname, 'rb') as f:
    dims = f.read(c_intsize*3)
    vecs = f.read()
  
  dims = np.frombuffer(dims, dtype=np.int32)
  vecs = np.frombuffer(vecs, dtype=np.float32)
  vecs = vecs.reshape([*dims, 3])

  return vecs

def write_vec(fname, xdim, ydim, zdim, vecs:np.ndarray, dtype=np.float32):
  grid_size = np.array([xdim, ydim, zdim], dtype=np.int32)
  with open(fname, 'wb') as f:
    f.write(grid_size.tobytes())
    f.write(vecs.astype(dtype).tobytes())
    
# grid file format: 3 int + floats
# xdim ydim zdim
# x0 , .... , x_xdim
# y0 , .... , x_ydim
# z0 , .... , x_zdim
def write_grid(fname, x, y, z, dtype=np.float32):
  grid_size = np.array([len(x), len(y), len(z)], dtype=np.int32)
  with open(fname, 'wb') as f:
    f.write(grid_size.tobytes())
    f.write(x.astype(dtype).tobytes())
    f.write(y.astype(dtype).tobytes())
    f.write(z.astype(dtype).tobytes())

def read_grid(fname):
  c_intsize = 4
  c_floatsize = 4
  with open(fname, 'rb') as f:
    dims = f.read(c_intsize*3)
    dims = np.frombuffer(dims, dtype=np.int32)
    x = f.read(c_floatsize*dims[0])
    y = f.read(c_floatsize*dims[1])
    z = f.read(c_floatsize*dims[2])
  x = np.frombuffer(x, dtype=np.float32)
  y = np.frombuffer(y, dtype=np.float32)
  z = np.frombuffer(z, dtype=np.float32)

  return [x, y, z]

# get all files with \d{postfix} in dir
# and return their zero-leading names

def get_zeroled_names(dir, postfix, num_zero):
    # rename single timestep files to 0 prepended e.g.: 1.vec -> 01.vec
  
  files = os.listdir(dir)
  digit_re = re.compile(rf"\d{postfix}")
  original_files = list(filter(digit_re.match, files))
  zeroled_files = [f"{'0'*num_zero}{name}" for name in original_files]

  original_paths = [os.path.join(dir, name) for name in original_files]
  zeroled_paths = [os.path.join(dir, name) for name in zeroled_files]

  return original_paths, zeroled_paths