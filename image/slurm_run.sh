#!/bin/bash
#SBATCH --job-name=tsam
#SBATCH --output=./slurm_vit_dtd/sam_uniform_effect_radius/%j.out
#SBATCH --error=./slurm_vit_dtd/sam_uniform_effect_radius/%j.err
#SBATCH --partition=xxx
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-59

num_samples=(3 5)
lr=(0.0003 0.001 0.003)
t=(0 1 5 20 100 400)
alpha=(0.001 0.01 0.1)
rho=(0.1)
eps=(0.005 0.01 0.05 0.1 0.5)
schedule_t=(0 1 2)
radius=(10 100)
scaling=(0.0001 0.00001 0.001 0.01 0.1 1 5)

len_blr=${#lr[@]}
len_num_samples=${#num_samples[@]}
len_t=${#t[@]}
len_alpha=${#alpha[@]}
len_rho=${#rho[@]}
len_eps=${#eps[@]}
len_radius=${#radius[@]}
len_schedule_t=${#schedule_t[@]}
len_scaling=${#scaling[@]}

let i=$SLURM_ARRAY_TASK_ID%$len_blr
let j=($SLURM_ARRAY_TASK_ID/$len_blr)%$len_rho
# let j=($SLURM_ARRAY_TASK_ID/$len_blr)%$len_num_samples
let k=($SLURM_ARRAY_TASK_ID/$((len_blr*$len_rho)))%$len_t
let r=($SLURM_ARRAY_TASK_ID/$((len_blr*$len_rho*$len_t)))%$len_radius
#let s=($SLURM_ARRAY_TASK_ID/$((len_blr*$len_rho*$len_t)))%$len_scaling


# finetune
python train_tsam.py --sampling random2 --learning_rate ${lr[$i]} --num_samples 40 --rho ${rho[$j]} \
--tilt ${t[$k]} --radius ${radius[$r]} --model vit --dataset dtd --schedule_t 0 --corruption 0 --epochs 20 \
--epsilon uniform

