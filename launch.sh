#!/bin/bash
#SBATCH --job-name=CR_GAN
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/output.out


python CR_GAN.py 20 1 3000