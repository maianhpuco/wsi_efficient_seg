#!/bin/bash
#SBATCH --job-name=ngoc/stitching
#SBATCH --output=ngoc_stitching.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

python tools/kpis/test_effvit.py --config configs/kpis.yaml --train_test_val train