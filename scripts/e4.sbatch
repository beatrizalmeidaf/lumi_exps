#!/bin/bash
#SBATCH -J lumi-e4
#SBATCH -p h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -o /home/user_beatrizalmeida/neuroips/lumi/logs/e4_%j.log
# E4: onde o treino em PT mudou os pesos e se a mudanca cabe num adaptador de posto baixo.
# posto 0 = Lumi com as Linear do pai (so normas/embeddings/Mamba A,D,conv do Lumi): controle
set -uo pipefail
cd /home/user_beatrizalmeida/neuroips/lumi
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY scripts/delta.py --ranks 0 16 64 256
for R in 0 16 64 256; do
  T=nemo4b__plus-delta-r$R; C=$LUMI_CKPTS/$T
  [ -f $LUMI_RESULTS/e1/eval/$T.parquet ] || $PY scripts/eval_mc.py --model $C --tag $T --gpu-mem 0.6
  rm -f $C/model*.safetensors
done
echo "=== fim $(date)"
