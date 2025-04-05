#!/bin/bash
#SBATCH --job-name=train_wsi_patch
#SBATCH --output=train_wsi_patch.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

python src/datasets/kpis/wsi_patch_dataset.py --config configs/kpis.yaml --train_test_val train