#!/bin/bash
#SBATCH -J StandardJob           
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH -p standard
#SBATCH --gres=gpu:1             
#SBATCH --tmp=5G

conda init
# Run pytorch via python environment
conda activate deformable_detr

cd /home/s478603/pos_estimation/Deformable-DETR/models/ops

echo 'export TORCH_CUDA_ARCH_LIST="8.6"' >> ~/.bashrc
source ~/.bashrc


sh ./make.sh
