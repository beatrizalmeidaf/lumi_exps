#!/bin/bash
#SBATCH -J lumi-e3
#SBATCH -p h100n3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o /home/user_beatrizalmeida/neuroips/lumi/logs/e3_%j.log
# E3: precisao mista, atencao (4 camadas, 4% dos pesos lineares) em BF16 e o resto quantizado.
#   real: W4A16 cal. PT sem a atencao (qualidade + velocidade no vLLM)
#   simulado: GPTQ W3 cal. PT/EN e RTN W4 sem a atencao (qualidade + TV)
set -uo pipefail
cd /home/user_beatrizalmeida/neuroips/lumi
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=scripts; R=$LUMI_CKPTS/real
/raid/$USER/lumi_tools/venv_llmc/bin/python $S/make_w4a16.py --base $LUMI --cal pt --skip-attn
T=lumi4b__real-w4a16-calpt-skipattn
[ -f $LUMI_RESULTS/e1/eval/$T.parquet ] || $PY $S/eval_mc.py --model $R/lumi4b__w4a16-calpt-skipattn --tag $T --gpu-mem 0.6
[ -f $LUMI_RESULTS/e2/bench/w4a16-skipattn/c128.json ] || bash $S/bench_serve.sh w4a16-skipattn $R/lumi4b__w4a16-calpt-skipattn
for SP in gptq-w3g128-calpt-skipattn gptq-w3g128-calen-skipattn rtn-w4g128-skipattn rtn-w3g128-skipattn; do
  T=lumi4b__$SP; C=$LUMI_CKPTS/$T
  [ -f $LUMI_RESULTS/e1/tv/$T.parquet ] && continue
  $PY $S/quantize.py --base $LUMI $SP
  [ -f $LUMI_RESULTS/e1/eval/$T.parquet ] || $PY $S/eval_mc.py --model $C --tag $T --gpu-mem 0.6
  $PY $S/tv_lang.py --parent $LUMI --comp $C --tags $T --bs 1
  rm -f $C/model*.safetensors
done
echo "=== fim $(date)"
