#!/bin/bash
#SBATCH --job-name=datasets_wsi
#SBATCH --output=datasets_wsi.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

python src/datasets/kpis/wsi_level.py --config configs/kpis.yaml --train_test_val train