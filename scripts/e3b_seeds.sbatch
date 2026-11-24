#!/bin/bash
#SBATCH -J lumi-e3b
#SBATCH -p h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /home/user_beatrizalmeida/neuroips/lumi/logs/e3b_%j.log
# E3b: variancia entre execucoes do W4A16 real (semente da amostragem de calibracao), PT e EN, com e sem atencao.
set -uo pipefail
cd /home/user_beatrizalmeida/neuroips/lumi
source env.sh
R=$LUMI_CKPTS/real; LLMC=/raid/$USER/lumi_tools/venv_llmc/bin/python
for SEED in 1 2; do
  for CFG in "pt" "en" "pt --skip-attn"; do
    set -- $CFG; C=$1; SK=${2:+-skipattn}
    $LLMC scripts/make_w4a16.py --base $LUMI --cal $C ${2:-} --seed $SEED
    N=lumi4b__w4a16-cal$C$SK-s$SEED
    [ -f $LUMI_RESULTS/e1/eval/lumi4b__real-w4a16-cal$C$SK-s$SEED.parquet ] || \
      $PY scripts/eval_mc.py --model $R/$N --tag lumi4b__real-w4a16-cal$C$SK-s$SEED --gpu-mem 0.6
    rm -f $R/$N/model*.safetensors
  done
done
echo "=== fim $(date)"
