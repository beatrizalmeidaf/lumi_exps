#!/usr/bin/env bash
#SBATCH --job-name=lumi-d4
#SBATCH --partition=h100n3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# d4_conformal: ver decisao/README.md do lumi_exps. Container + venv do lumi-decision (implementação nativa).
# Submeter de /raid/$USER/lumi-decision (os logs vão para logs/ de lá).
set -euo pipefail
cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
OUT="$HOME/neuroips/lumi/decisao/resultados/d4"; mkdir -p "$OUT"
lumi_exec python "$HOME/neuroips/lumi/decisao/lumi_extra.py" conformal --runs "/raid/$USER/lumi-decision/runs" --prefix "${1:-e1_lumi_27b30f4_nativa}" --out "$OUT"
