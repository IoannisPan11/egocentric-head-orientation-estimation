#!/bin/bash
#SBATCH --job-name=resize_head
#SBATCH --output=resize_head_%j.out
#SBATCH --error=resize_head_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

/home/e12507305/micromamba/envs/thesis/bin/python resize_head_mounted.py
