#!/bin/bash
#SBATCH --job-name=SAM_REPRODUCE
#SBATCH --output=/scratch/bcom/aananth2/SAM-2nd-order/experiments/SAM_WRN_GA_3_avg_sharpness%x-%A_%a.out
#SBATCH --error=/scratch/bcom/aananth2/SAM-2nd-order/experiments/SAM_WRN_GA_3_avg_sharpness%x-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=200G
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user=aananth2@illinois.edu
#SBATCH --account=bcom-delta-gpu

source /u/aananth2/miniconda3/bin/activate
conda init bash
conda activate sam_env

cd /scratch/bcom/aananth2/SAM-2nd-order/experiments/

# python train.py --optim sam --steps 5
python train.py --optim sam --steps 3
python train.py --optim sam --steps 3
python train.py --optim sam --steps 3
python train.py --optim sam --steps 3
python train.py --optim sam --steps 3

source deactivate