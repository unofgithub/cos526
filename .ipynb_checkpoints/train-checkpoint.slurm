#!/bin/bash
#SBATCH --job-name=train-point-rad   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1                   # number of GPUs your job requests
##SBATCH --constraint=a100
#SBATCH --mem=50G                      # amount of memory needed
#SBATCH -o outputs.txt            # send stdout to outfile
#SBATCH -e errfiles.txt           # send stderr to errfile
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=nobliney@princeton.edu

module purge
module load anaconda3/2022.5
conda activate pytorch3d

python main.py --datadir /scratch/network/by7705/cos526/nerf_synthetic --dataname materials --basedir /scratch/network/by7705/cos526/point-radiance/logs --data_r 0.012 --splatting_r 0.015 --use_msssim True