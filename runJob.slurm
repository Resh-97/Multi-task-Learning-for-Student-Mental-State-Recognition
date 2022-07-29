#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=TRAIN 
#SBATCH -p lyceum
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output =output3.out
#SBATCH --error =error3.err


source activate project

python FECNet.py
