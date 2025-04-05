#!/bin/bash
#SBATCH --job-name=wsi_patch
#SBATCH --output=wsi_patch.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module load python/3.10
module load conda
conda init 
conda activate efficientnet
python experiments/processing/kpis_patching.py --config configs/kpis.yaml

conda deactivate