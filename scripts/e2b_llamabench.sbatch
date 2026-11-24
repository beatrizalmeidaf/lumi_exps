#!/bin/bash
#SBATCH -J lumi-e2b
#SBATCH -p h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o /home/user_beatrizalmeida/neuroips/lumi/logs/e2b_%j.log
# E2b: so a velocidade do llama.cpp (llama-bench), que o E2 pulou por um erro no kill do servidor.
set -uo pipefail
cd /home/user_beatrizalmeida/neuroips/lumi
source env.sh
echo "=== GPU $CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader) $(date)"
LC=/raid/$USER/lumi_tools/llama.cpp/build/bin; G=/raid/$USER/lumi_tools/gguf; OUT=$LUMI_RESULTS/e2/gguf
for V in bf16 q8_0 q4km q4km-imaten q4km-imatpt; do
  $LC/llama-bench -m $G/lumi4b-$V.gguf -p 512 -n 256 -ngl 99 -r 5 -o json > $OUT/bench_$V.json
done
echo "=== fim $(date)"
