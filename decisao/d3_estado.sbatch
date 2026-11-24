#!/usr/bin/env bash
#SBATCH --job-name=lumi-d3
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# d3_estado: ver decisao/README.md do lumi_exps. Container + venv do lumi-decision (implementação nativa).
# Submeter de /raid/$USER/lumi-decision (os logs vão para logs/ de lá).
set -euo pipefail
cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
OUT="$HOME/neuroips/lumi/decisao/resultados/d3"; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.used --format=csv
lumi_exec python "$HOME/neuroips/lumi/decisao/lumi_extra.py" estado --out "$OUT" "$@"
