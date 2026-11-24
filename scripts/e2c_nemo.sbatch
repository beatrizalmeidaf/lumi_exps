#!/bin/bash
#SBATCH -J lumi-e2c
#SBATCH -p h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o /home/user_beatrizalmeida/neuroips/lumi/logs/e2c_%j.log
# E2c: refaz a velocidade da Nemotron base com a mesma amostragem do Lumi (top_p 1). O generation_config
# da Nemotron traz top_p 0.95, que liga outro amostrador no vLLM e compila kernels durante a medicao.
set -uo pipefail
cd /home/user_beatrizalmeida/neuroips/lumi
source env.sh
bash scripts/bench_serve.sh nemo-bf16 $NEMO --generation-config vllm
