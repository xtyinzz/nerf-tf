#!/bin/bash
#
#SBATCH --job-name=nerf_
#SBATCH --output="log/nyx500/osc_nyx500_high_%j.out"
#SBATCH --signal=USR1@30
#
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --time=23:59:00
#
#SBATCH --account=PAS0027

# find project dir
source $HOME/loadvis.bashrc
cd $HOME/project/nerf-tf

# prep software
source activate nerf

# execute job
srun python run_nerf.py --config=cfg/nyx.yaml
# srun python train_block.py --nncfg=nn_block_half.yaml --resume
# python test.py --testcfg=cfg/nntest_curv.yaml
# python mem.py