#!/bin/bash
#SBATCH --job-name=train_wsi_patch
#SBATCH --output=train_wsi_patch.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4


python experiments/processing/wsi_level.py --config configs/kpis.yaml

conda deactivate