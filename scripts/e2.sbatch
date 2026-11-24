#!/bin/bash
#SBATCH -J lumi-e2
#SBATCH -p h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /home/user_beatrizalmeida/neuroips/lumi/logs/e2_%j.log
# E2: Lumi em formatos reais (vLLM: BF16, FP8, FP8+KV FP8, W4A16 PT/EN; llama.cpp: GGUF).
# Qualidade com a avaliacao do E1 e velocidade com o protocolo da sparkbench. Idempotente.
set -uo pipefail
cd /home/user_beatrizalmeida/neuroips/lumi
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=scripts
R=$LUMI_CKPTS/real
LLMC=/raid/$USER/lumi_tools/venv_llmc/bin/python
echo "=== GPU $CUDA_VISIBLE_DEVICES $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader) $(date)"

# 1. W4A16 real (GPTQ do llm-compressor), calibracao PT e EN
for C in pt en; do $LLMC $S/make_w4a16.py --base $LUMI --cal $C; done

# 1b. simulado com o mesmo grupo do W4A16 real (64), para checar simulado x real
$PY $S/quantize.py --base $LUMI gptq-w4g64-calpt
$PY $S/eval_mc.py --model $LUMI_CKPTS/lumi4b__gptq-w4g64-calpt --tag lumi4b__gptq-w4g64-calpt --gpu-mem 0.6

# 2. qualidade com kernels reais (BF16 ja esta no E1)
$PY $S/eval_mc.py --model $LUMI --tag lumi4b__real-fp8 --quantization fp8 --gpu-mem 0.6
$PY $S/eval_mc.py --model $LUMI --tag lumi4b__real-fp8kv --quantization fp8 --kv-cache-dtype fp8 --gpu-mem 0.6
for C in pt en; do
  $PY $S/eval_mc.py --model $R/lumi4b__w4a16-cal$C --tag lumi4b__real-w4a16-cal$C --gpu-mem 0.6
done

# 3. velocidade (mesmo --gpu-memory-utilization em todas as variantes)
bash $S/bench_serve.sh bf16 $LUMI
bash $S/bench_serve.sh fp8 $LUMI --quantization fp8
bash $S/bench_serve.sh fp8kv $LUMI --quantization fp8 --kv-cache-dtype fp8
bash $S/bench_serve.sh w4a16 $R/lumi4b__w4a16-calpt
bash $S/bench_serve.sh nemo-bf16 $NEMO

# 4. GGUF
bash $S/e2_gguf.sh
echo "=== fim $(date)"
