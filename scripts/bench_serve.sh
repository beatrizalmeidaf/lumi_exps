#!/bin/bash
# Bancada de servico (E2) com o protocolo da sparkbench do CEIA:
#   512 tokens de entrada -> 256 de saida (ignore_eos), carga fechada de 1 a 128 usuarios.
# Cliente: `vllm bench serve` (validado a +-7% contra a sparkbench no relatorio da RTX 3060).
# Potencia: nvidia-smi a cada 200 ms durante cada nivel -> tokens/J.
#
# Uso: bash lumi/scripts/bench_serve.sh <nome> <modelo> [args extras do vllm serve...]
# Saida: lumi/results/e2/bench/<nome>/c<N>.json, power_c<N>.csv, server.log
set -uo pipefail
NAME=$1; MODEL=$2; shift 2
cd "$(dirname "$0")/.."
source env.sh
OUT=$LUMI_RESULTS/e2/bench/$NAME
mkdir -p $OUT
GPU=${CUDA_VISIBLE_DEVICES:-0}
PORT=${PORT:-8199}
LEVELS=${LEVELS:-"1 2 4 8 16 32 64 128"}

if [ -f $OUT/c128.json ] && [ -z "${FORCE:-}" ]; then echo "[skip] $NAME"; exit 0; fi

vllm serve $MODEL --port $PORT --host 127.0.0.1 --max-model-len 4096 --max-num-seqs 128 \
  --gpu-memory-utilization ${GPU_MEM:-0.85} --seed 0 \
  --default-chat-template-kwargs '{"enable_thinking": false}' "$@" > $OUT/server.log 2>&1 &
SRV=$!
trap "kill $SRV 2>/dev/null; wait $SRV 2>/dev/null" EXIT
for i in $(seq 600); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "servidor morreu"; tail -30 $OUT/server.log; exit 1; }
  sleep 2
done
echo "[up] $NAME $(date +%T)"

bench() {  # $1 = concorrencia, $2 = n de prompts, $3 = arquivo
  vllm bench serve --backend vllm --model $MODEL --port $PORT --host 127.0.0.1 \
    --dataset-name random --random-input-len 512 --random-output-len 256 --random-range-ratio 0 \
    --ignore-eos --max-concurrency $1 --num-prompts $2 --request-rate inf --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --goodput ttft:1000 tpot:50 \
    --save-result --result-dir $OUT --result-filename $3 > $OUT/${3%.json}.txt 2>&1
}
bench 8 32 warmup.json

for C in $LEVELS; do
  N=$(( C * 4 < 32 ? 32 : C * 4 ))
  nvidia-smi -i $GPU --query-gpu=timestamp,power.draw,utilization.gpu,clocks.sm,memory.used \
    --format=csv,noheader,nounits -lms 200 > $OUT/power_c$C.csv &
  SMI=$!
  bench $C $N c$C.json
  kill $SMI
  echo "[c=$C] $(grep -E 'Output token throughput' $OUT/c$C.txt | head -1)"
done
grep -E "KV cache|Maximum concurrency|GPU KV cache size|model weights took|Loading weights took" $OUT/server.log | head -8
