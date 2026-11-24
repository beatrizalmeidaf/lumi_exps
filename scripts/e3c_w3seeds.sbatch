#!/bin/bash
#SBATCH -J lumi-e3c
#SBATCH -p h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /home/user_beatrizalmeida/neuroips/lumi/logs/e3c_%j.log
# E3c: o efeito da lingua da calibracao em 3 bits (achado principal) com outra semente de calibracao,
# no Lumi e na Nemotron.
set -uo pipefail
cd /home/user_beatrizalmeida/neuroips/lumi
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for BASE in $LUMI $NEMO; do
  B=$($PY -c "import sys;sys.path.insert(0,'scripts');from quantize import short;print(short('$BASE'))")
  S=${1:-1}
  for SP in gptq-w3g128-calpt-s$S gptq-w3g128-calen-s$S; do
    T=${B}__$SP; C=$LUMI_CKPTS/$T
    [ -f $LUMI_RESULTS/e1/eval/$T.parquet ] && continue
    $PY scripts/quantize.py --base $BASE $SP
    $PY scripts/eval_mc.py --model $C --tag $T --gpu-mem 0.6
    rm -f $C/model*.safetensors
  done
done
echo "=== fim $(date)"
