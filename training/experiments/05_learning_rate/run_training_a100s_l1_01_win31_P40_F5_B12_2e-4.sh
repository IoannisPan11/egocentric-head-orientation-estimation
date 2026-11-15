#!/bin/bash
#SBATCH --job-name=2e-4_B12_P40_F5_win31_l1_b01
#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:a100s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/e12507305/micromamba/envs/thesis-gpu/logs/2e-4_B12_P40_F5_win31_l1_b01%j.out
#SBATCH --error=/home/e12507305/micromamba/envs/thesis-gpu/logs/2e-4_B12_P40_F5_win31_l1_b01%j.err

export TMPDIR=/home/e12507305/tmp
mkdir -p "$TMPDIR"
mkdir -p /home/e12507305/micromamba/envs/thesis-gpu/logs

echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "TMPDIR: $TMPDIR"
echo "GPU info:"
nvidia-smi

cd /home/e12507305/micromamba/envs/thesis-gpu

echo "Working directory: $(pwd)"

echo "Python path from thesis-gpu env:"
micromamba run -n thesis-gpu which python

echo "Torch / CUDA check:"
micromamba run -n thesis-gpu python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

echo "Starting training..."
micromamba run -n thesis-gpu python -u train_subset_l1_01_win31_P40_F5_B12_2e-4.py