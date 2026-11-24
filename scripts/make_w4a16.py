"""Checkpoint W4A16 REAL (compressed-tensors, kernels Marlin no vLLM) com GPTQ do llm-compressor.

Roda no venv do llm-compressor ($LUMI_TOOLS/venv_llmc), nao no do vLLM.
Mesma receita do E1 (GPTQ, 4 bits, grupo 128, 128 x 512 tokens de calibracao), mudando so o
idioma da calibracao: FineWeb-2 por_Latn (pt) ou C4 (en). lm_head e embeddings ficam em bf16.

Uso: python make_w4a16.py --base ceia-research/lumi-4b --cal pt
Saida: $LUMI_CKPTS/real/<curto>__w4a16-cal<pt|en>/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset

DATA = Path(os.environ["LUMI_DATA"])
CKPTS = Path(os.environ["LUMI_CKPTS"]) / "real"
SHORT = {"ceia-research/lumi-4b": "lumi4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": "nemo4b"}


def calib_ds(tok, lang: str, n: int = 128, L: int = 512, seed: int = 0) -> Dataset:
    """Mesma amostragem do E1 (quantize.calibration): um documento por janela de L tokens."""
    texts = pd.read_parquet(DATA / f"web_{lang}.parquet")["text"].tolist()
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in torch.randperm(len(texts), generator=g).tolist():
        ids = tok(texts[i], add_special_tokens=False).input_ids
        if len(ids) > L:
            s = int(torch.randint(0, len(ids) - L, (1,), generator=g))
            rows.append([tok.bos_token_id] + ids[s:s + L - 1])
        if len(rows) == n:
            break
    return Dataset.from_dict({"input_ids": rows, "attention_mask": [[1] * L for _ in rows]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cal", choices=["pt", "en"], required=True)
    ap.add_argument("--seed", type=int, default=0, help="semente da amostragem da calibracao (variancia entre execucoes)")
    ap.add_argument("--skip-attn", action="store_true", help="precisao mista: q/k/v/o_proj ficam em BF16")
    args = ap.parse_args()
    out = CKPTS / f"{SHORT.get(args.base, args.base.split('/')[-1])}__w4a16-cal{args.cal}{'-skipattn' if args.skip_attn else ''}{f'-s{args.seed}' if args.seed else ''}"
    if (out / "lumi_meta.json").exists():
        print(f"[skip] {out}")
        return

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    # hidden_size 3136 = 64 x 49 nao e multiplo de 128: grupo 64 divide todas as Linear (3136, 7680, 12544)
    # simetrico, como o preset W4A16 (formato padrao dos checkpoints publicados; Marlin no vLLM)
    ign = ["lm_head"] + (["re:.*\\.(q|k|v|o)_proj$"] if args.skip_attn else [])
    recipe = GPTQModifier(ignore=ign, dampening_frac=0.01, config_groups={"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 64}}})
    oneshot(model=model, dataset=calib_ds(tok, args.cal, seed=args.seed), recipe=recipe, max_seq_length=512,
            num_calibration_samples=128, pipeline=os.environ.get("LLMC_PIPELINE", "sequential"),
            batch_size=1)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    meta = {"base": args.base, "cal": args.cal, "scheme": "W4A16 g64 simetrico, GPTQ (llm-compressor)",
            "seed": args.seed, "skip_attn": args.skip_attn, "seconds": round(time.time() - t0, 1)}
    (out / "lumi_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {out} {meta}")


if __name__ == "__main__":
    main()
