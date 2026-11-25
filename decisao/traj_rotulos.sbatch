#!/usr/bin/env bash
#SBATCH --job-name=lumi-traj-rot
#SBATCH --partition=h100n3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# T (trajetória pelas camadas) e R (rótulos para família nova); protocolos em decisao/README.md. Só CPU.
# Submeter de /raid/$USER/lumi-decision:  sbatch ~/neuroips/lumi/decisao/traj_rotulos.sbatch
set -uo pipefail
cd "/raid/$USER/lumi-decision"
source slurm/config.sh
D="$HOME/neuroips/lumi/decisao"
rc=0
lumi_exec python "$D/trajetoria.py" traj --dump "$D/resultados/d2" --out "$D/resultados/traj" || rc=1
lumi_exec python "$D/trajetoria.py" rotulos --runs "/raid/$USER/lumi-decision/runs" \
  --prefix e1_lumi_27b30f4_nativa --out "$D/resultados/rotulos" || rc=1
exit $rc
