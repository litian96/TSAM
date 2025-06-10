#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=./slurm/eval/sgd/%j.out
#SBATCH --error=./slurm/eval/sgd/%j.err
#SBATCH --partition=xxx
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-0

num_samples=(3)
lr=(0.0003 0.001 0.003 0.01)
t=(0 1 5 20 100 400)
alpha=(0.001 0.01 0.1)
rho=(0.1)
eps=(0.005 0.0075 0.01 0.015 0.02 0.035 0.05 0.075 0.1)

len_blr=${#lr[@]}
len_num_samples=${#num_samples[@]}
len_t=${#t[@]}
len_alpha=${#alpha[@]}
len_rho=${#rho[@]}
len_eps=${#eps[@]}


python eval.py --method sgd --dataset cifar100 --model resnet --eps 0.1
