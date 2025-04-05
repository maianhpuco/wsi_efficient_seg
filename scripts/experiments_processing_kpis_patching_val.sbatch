#!/bin/bash
#SBATCH --job-name=val_wsi_patch
#SBATCH --output=val_wsi_patch.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4


python experiments/processing/kpis_patching.py --config configs/kpis.yaml --train_test_val val

conda deactivate