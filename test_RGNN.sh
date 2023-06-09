#!/bin/bash
#
#SBATCH --mem=4G 
#SBATCH --output=/output/slurm.out
#SBATCH --error=/output/slurm.err
python3 test_RGNN.py