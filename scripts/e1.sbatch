#!/bin/bash
#SBATCH -J lumi-e1
#SBATCH -p h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /home/user_beatrizalmeida/neuroips/lumi/logs/e1_%j.log
# E1: quantizacao do Lumi e do Nemotron base, qualidade por idioma (MC) e TV por token.
# Idempotente: cada etapa pula o que ja existe.
set -euo pipefail
cd /home/user_beatrizalmeida/neuroips/lumi
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=scripts

SPECS="rtn-w4g128 rtn-w3g128 gptq-w4g128-calen gptq-w4g128-calpt gptq-w3g128-calen gptq-w3g128-calpt \
       rtn-w3g128-onlymamba rtn-w3g128-onlymlp rtn-w3g128-onlyattn"

for BASE in $LUMI $NEMO; do
  B=$($PY -c "import sys;sys.path.insert(0,'$S');from quantize import short;print(short('$BASE'))")
  echo "=== $B $(date)"
  $PY $S/eval_mc.py --model $BASE --tag ${B}__bf16 --gpu-mem 0.6
  for SP in $SPECS; do
    T=${B}__$SP; C=$LUMI_CKPTS/$T
    if [ -f $LUMI_RESULTS/e1/eval/$T.parquet ] && [ -f $LUMI_RESULTS/e1/tv/$T.parquet ]; then
      rm -f $C/model*.safetensors; continue; fi
    $PY $S/quantize.py --base $BASE $SP
    [ -f $LUMI_RESULTS/e1/eval/$T.parquet ] || $PY $S/eval_mc.py --model $C --tag $T --gpu-mem 0.6
    $PY $S/tv_lang.py --parent $BASE --comp $C --tags $T --bs 1
    # cota do /raid: o checkpoint simulado (7,5 GB) nao e mais necessario; fica so o lumi_meta.json
    rm -f $C/model*.safetensors
  done
done
echo "=== fim $(date)"
