#!/usr/bin/env bash
#SBATCH --job-name=lumi-sonda
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# P: sonda linear nos estados internos (decisao/README.md do lumi_exps). Extração na GPU, análise na CPU.
# Submeter de /raid/$USER/lumi-decision.  Pular a extração já feita: sbatch ... sonda.sbatch analisar
set -euo pipefail
cd "/raid/$USER/lumi-decision"
source slurm/config.sh
export HF_HUB_OFFLINE=1
D="$HOME/neuroips/lumi/decisao"; F="/raid/$USER/lumi_sonda"
if [ "${1:-tudo}" != "analisar" ]; then
  nvidia-smi --query-gpu=name,memory.used --format=csv
  lumi_exec python "$D/sonda.py" extrair --out "$F"
fi
lumi_exec python "$D/sonda.py" analisar --feats "$F" --out "$D/resultados/sonda"
