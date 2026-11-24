#!/bin/bash
# E2, parte llama.cpp: GGUF do Lumi em BF16, Q8_0 e Q4_K_M (sem imatrix, imatrix EN, imatrix PT).
#   qualidade: KLD/top-1 contra o BF16 em texto PT e EN (llama-perplexity) + multipla escolha via llama-server
#   velocidade: llama-bench (pp512 / tg256, 1 usuario)
# Idempotente. Uso: bash lumi/scripts/e2_gguf.sh   (com CUDA_VISIBLE_DEVICES definido)
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
T=/raid/$USER/lumi_tools
LC=$T/llama.cpp/build/bin
G=$T/gguf; mkdir -p $G
OUT=$LUMI_RESULTS/e2/gguf; mkdir -p $OUT
SNAP=$(ls -d $HF_HUB_CACHE/models--ceia-research--lumi-4b/snapshots/* | head -1)

# textos: imatrix dos documentos [0, 1000), avaliacao dos [1000, ...) (os mesmos da TV do E1)
for L in pt en; do
  [ -f $G/imat_$L.txt ] || $PY -c "
import pandas as pd; t=pd.read_parquet('$LUMI_DATA/web_$L.parquet')['text'].tolist()
open('$G/imat_$L.txt','w').write('\n\n'.join(t[:1000])); open('$G/eval_$L.txt','w').write('\n\n'.join(t[1000:1600]))"
done

[ -f $G/lumi4b-bf16.gguf ] || $T/venv_llmc/bin/python $T/llama.cpp/convert_hf_to_gguf.py $SNAP \
  --outtype bf16 --outfile $G/lumi4b-bf16.gguf
for L in pt en; do
  [ -f $G/imatrix_$L.gguf ] || $LC/llama-imatrix -m $G/lumi4b-bf16.gguf -f $G/imat_$L.txt -o $G/imatrix_$L.gguf \
    -c 512 --chunks 128 -ngl 99 > $OUT/imatrix_$L.log 2>&1
done
[ -f $G/lumi4b-q8_0.gguf ]       || $LC/llama-quantize $G/lumi4b-bf16.gguf $G/lumi4b-q8_0.gguf Q8_0 > $OUT/quantize_q8_0.log 2>&1
[ -f $G/lumi4b-q4km.gguf ]       || $LC/llama-quantize $G/lumi4b-bf16.gguf $G/lumi4b-q4km.gguf Q4_K_M > $OUT/quantize_q4km.log 2>&1
for L in pt en; do
  [ -f $G/lumi4b-q4km-imat$L.gguf ] || $LC/llama-quantize --imatrix $G/imatrix_$L.gguf $G/lumi4b-bf16.gguf \
    $G/lumi4b-q4km-imat$L.gguf Q4_K_M > $OUT/quantize_q4km-imat$L.log 2>&1
done
ls -la $G/*.gguf

VARS="q8_0 q4km q4km-imaten q4km-imatpt"
# KLD contra o BF16 (logits base salvos uma vez por idioma)
for L in pt en; do
  [ -f $G/base_$L.kld ] || $LC/llama-perplexity -m $G/lumi4b-bf16.gguf -f $G/eval_$L.txt -c 512 --chunks 128 \
    -ngl 99 --kl-divergence-base $G/base_$L.kld > $OUT/ppl_bf16_$L.log 2>&1
  for V in $VARS; do
    [ -f $OUT/kld_${V}_$L.log ] || $LC/llama-perplexity -m $G/lumi4b-$V.gguf -c 512 --chunks 128 -ngl 99 \
      --kl-divergence-base $G/base_$L.kld --kl-divergence > $OUT/kld_${V}_$L.log 2>&1
  done
done

# multipla escolha via llama-server (MMLU pareado 2000 itens por idioma + provas completas)
PORT=8299
for V in bf16 $VARS; do
  TAG=lumi4b__gguf-$V
  [ -f $LUMI_RESULTS/e1/eval/$TAG.parquet ] && continue
  $LC/llama-server -m $G/lumi4b-$V.gguf --port $PORT -ngl 99 -c 65536 -np 16 --jinja > $OUT/server_$V.log 2>&1 &
  SRV=$!
  until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do sleep 2; kill -0 $SRV; done
  $PY scripts/eval_mc.py --model $G/lumi4b-$V.gguf --tag $TAG --api http://127.0.0.1:$PORT --mmlu-n 2000 --workers 16 || true
  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
done

# velocidade com 1 usuario (so em GPU exclusiva; ONLY_QUALITY=1 pula)
[ -n "${ONLY_QUALITY:-}" ] && { echo "[ok] gguf (so qualidade)"; exit 0; }
for V in bf16 $VARS; do
  [ -f $OUT/bench_$V.json ] || $LC/llama-bench -m $G/lumi4b-$V.gguf -p 512 -n 256 -ngl 99 -r 5 -o json > $OUT/bench_$V.json
done
echo "[ok] gguf"
