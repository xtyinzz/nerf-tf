import os, sys
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import imageio
import pprint

import matplotlib.pyplot as plt

import run_nerf
import run_nerf_helpers
from data.load_nyx import load_nyx_data
from data.vis_io import read_raw_vti, write_vti
import vtk

### Load trained network weights

basedir = './logs'
expname = 'nyx_coldhot'
ft_str = '' 
ft_str = '--ft_path {}'.format(os.path.join(basedir, expname, 'model_130000.npy'))
# output field names
fpath = "vol/nyx_fine_box_coldhot"
raw_fpath = fpath + ".raw"

config = os.path.join(basedir, expname, 'config.txt')
print('Args:')
print(open(config, 'r').read())

parser = run_nerf.config_parser()

args = parser.parse_args('--config {} '.format(config) + ft_str)

# Create nerf model
_, render_kwargs_test, start, grad_vars, models = run_nerf.create_nerf(args)


images, poses, hw, nearFar, bbox, iSplit, render_poses = load_nyx_data(
  args.datadir, testSplit=args.test_split
)
# nears, fars = nearFar[:,0], nearFar[:,1]
near, far = np.mean(nearFar, 0) / 255 * 2 - 1
# near = float(near)
# far = float(far)
focal = hw[0] / np.tan(np.deg2rad(30/2))
hwf = [*hw, focal]


bds_dict = {
    'near' : tf.cast(near, tf.float32),
    'far' : tf.cast(far, tf.float32),
}
render_kwargs_test.update(bds_dict)

print('Render kwargs:')
pprint.pprint(render_kwargs_test)

net_fn = render_kwargs_test['network_query_fn']
print(net_fn)

# Render an overhead view to check model was loaded correctly
# c2w = np.eye(4)[:3,:4].astype(np.float32) # identity pose matrix
# c2w[2,-1] = 4.
H, W, focal = hwf
# down = 8
# test = run_nerf.render(H//down, W//down, focal/down, c2w=c2w, **render_kwargs_test)
# img = np.clip(test[0],0,1)
# plt.imshow(img)
# plt.show()


### Query network on dense 3d grid of points
N = 255
dimmax = 0.6*(234/255) # visualization found 0.6*234=0.5506 to be the bbox for volume
t = np.linspace(-dimmax, dimmax, N+1)
descend_t = np.linspace(dimmax, -dimmax, N+1)

query_pts = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).astype(np.float32)
print(query_pts.shape)
sh = query_pts.shape
# query_pts = query_pts.transpose(0, 2)
query_pts = query_pts.swapaxes(0, 2)
flat = query_pts.reshape([-1,3])


def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret
    
    
fn = lambda i0, i1 : net_fn(flat[i0:i1,None,:], viewdirs=np.zeros_like(flat[i0:i1]), network_fn=render_kwargs_test['network_fine']) # fine model output
# fn = lambda i0, i1 : net_fn(flat[i0:i1,None,:], viewdirs=np.zeros_like(flat[i0:i1]), network_fn=render_kwargs_test['network_fn']) # coarse model output
chunk = 1024*64
raw = np.concatenate([fn(i, i+chunk).numpy() for i in range(0, flat.shape[0], chunk)], 0)
raw = np.reshape(raw, list(sh[:-1]) + [-1])
sigma = np.maximum(raw[...,-1], 0.)

# np.save("vol/lego.npy", raw)

sigma.tofile(raw_fpath)

bbox = np.array([
    [0,0,0],
    [N, N, N]
], dtype=np.float32)

sigma_vti = read_raw_vti(raw_fpath, "dnerf", bbox)
write_vti(fpath+".vti", sigma_vti)

# logFilter = vtk.vtkArrayCalculator()
# logFilter.SetInputData(sigma_vti)
# logFilter.AddScalarArrayName("dnerf")
# # logFilter.AddScalarVariable ("d_nerf", "d_nerf")
# print(logFilter.GetScalarArrayNames())
# logFilter.SetFunction("log10(dnerf)")
# logFilter.SetResultArrayName("logdnerf")
# logFilter.SetResultArrayType(vtk.VTK_FLOAT)
# logFilter.Update()
# lodd_sigma_vti = logFilter.GetOutput()
# write_vti(fpath+".vti", lodd_sigma_vti)