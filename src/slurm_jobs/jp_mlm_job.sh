#!/bin/bash -l
# Declaring Slurm Configuration Options
#SBATCH --job-name="jp_mlm_job"
#SBATCH --comment="FL job"
#SBATCH --account="fl-mlm"
#SBATCH --partition=tier3
#SBATCH --time=0-03:00:00

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:2
#SBATCH --gpus-per-task=a100:2

# Loading Software/Libraries

# Running Code
source ~/env/bin/activate
cd ../fl-mlm
flwr run . > flwr.log 2>&1
