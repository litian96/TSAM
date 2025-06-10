#!/bin/bash
#SBATCH --job-name=distillbert
#SBATCH --output=./slurm_bert/rte/tsam/%j.out
#SBATCH --error=./slurm_bert/rte/tsam/%j.err
#SBATCH --partition=xxx
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-11

num_samples=(3 5)
lr=(0.0003 0.001 0.003)
t=(0 1 5 20 100 400)
alpha=(0.001 0.01 0.1)
rho=(0.1)
eps=(0.005 0.01 0.05 0.1 0.5)


len_blr=${#lr[@]}
len_num_samples=${#num_samples[@]}
len_t=${#t[@]}
len_alpha=${#alpha[@]}
len_rho=${#rho[@]}
len_eps=${#eps[@]}


let i=$SLURM_ARRAY_TASK_ID%$len_blr
let j=($SLURM_ARRAY_TASK_ID/$len_blr)%$len_rho
# let j=($SLURM_ARRAY_TASK_ID/$len_blr)%$len_num_samples
let k=($SLURM_ARRAY_TASK_ID/$((len_blr*$len_rho)))%$len_t
#let r=($SLURM_ARRAY_TASK_ID/$((len_blr*$len_rho*$len_t)))%$len_radius


python train_sam.py --learning_rate ${lr[$i]} --model distillbert --rho ${rho[$j]} --dataset rte --corruption 0.0 --epochs 10




