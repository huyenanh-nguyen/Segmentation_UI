#!/bin/bash
#SBATCH --job-name=cellpose_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Email notifications setup:
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ng_huyenanh@mailbox.org

# 1. Load CUDA environment modules
module purge
module load gcc/11.2.0
module load gpu/cuda/11.8

# 2. Activate environment using full path

source /u/hanguy/Segmentation_UI/env/bin/activate

# 3. Run training script using full path

python /u/hanguy/Segmentation_UI/Segmentation/cellpose_train.py