#!/bin/bash
#SBATCH --job-name=resize_head
#SBATCH --output=resize_head_%j.out
#SBATCH --error=resize_head_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# 10_crop_and_resize_frames.sh
#
# SLURM job script for step 10 (crop and resize) on the TU Wien dataLAB cluster.
# Submit with: sbatch 10_crop_and_resize_frames.sh
# The Python interpreter is the original micromamba environment ("thesis") on the cluster.
# This step ran on the cluster; steps 00-09 ran on the Windows workstation.

/home/e12507305/micromamba/envs/thesis/bin/python 10_crop_and_resize_frames.py
