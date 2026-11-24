# source lumi/env.sh -- ambiente dos experimentos do Lumi (herda o do neuroips)
_LUMI_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$_LUMI_DIR/../env.sh"
export HF_HUB_CACHE=/raid/$USER/hf_extra          # Lumi e Nemotron base ficam aqui
export LUMI_DATA=/raid/$USER/lumi_data
export LUMI_CKPTS=/raid/$USER/dxs_ckpts/lumi      # pesos comprimidos (recriaveis)
export LUMI_RESULTS=$_LUMI_DIR/results
export LUMI=ceia-research/lumi-4b
export NEMO=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
mkdir -p $LUMI_CKPTS
