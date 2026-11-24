#!/usr/bin/env bash
#SBATCH --job-name=lumi-d1
#SBATCH --partition=h100n3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# D1: desacordo entre ordens como sinal de abstenção (protocolo em decisao/README.md do lumi_exps).
# Só CPU. Roda no container + venv do lumi-decision para usar lumi_decision.metrics do grupo.
# Submeter de /raid/$USER/lumi-decision:
#   sbatch --dependency=afterok:<E1> ~/neuroips/lumi/decisao/d1_analise.sbatch [prefixo]
set -euo pipefail
cd "/raid/$USER/lumi-decision"
source slurm/config.sh
PREFIX="${1:-e1_lumi_27b30f4_nativa}"
OUT="$HOME/neuroips/lumi/decisao/resultados/d1"
mkdir -p "$OUT"
lumi_exec python "$HOME/neuroips/lumi/decisao/ordem_abstencao.py" \
  --runs "/raid/$USER/lumi-decision/runs" --prefix "$PREFIX" --out "$OUT"
