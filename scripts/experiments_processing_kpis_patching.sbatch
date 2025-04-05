#!/bin/bash
#SBATCH --job-name=wsi_patch
#SBATCH --output=wsi_patch.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load python/3.10
module load conda
source /project/hnguyen2/mvu9/conda_envs/efficientnet/bin/activate

python src/datasets/kpis/extract_patches.py --config configs/kpis.yaml

conda deactivate