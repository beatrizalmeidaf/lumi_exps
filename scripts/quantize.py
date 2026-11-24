"""Checkpoints comprimidos SIMULADOS (pesos desquantizados em bf16) do Lumi e do Nemotron base.

Mesmo pipeline do GLOD/neuroips (scripts/make_compressed.py), com duas extensoes:
  * idioma da calibracao: C4 (en, como no GLOD) ou FineWeb-2 por_Latn (pt);
  * quantizar so um tipo de bloco do hibrido NemotronH: Mamba2 (in/out_proj),
    MLP (up/down_proj) ou atencao (q/k/v/o_proj).
Embeddings, lm_head, conv1d e parametros do SSM (A, D, dt) ficam em bf16, como nos
quantizadores usados na pratica.

Spec:  <metodo>-w<bits>g<grupo>[-cal<en|pt>][-<only|skip><mamba|mlp|attn>]
       metodo in {rtn, gptq, awq}; rtn ignora -cal.
Ex.:   rtn-w4g128   gptq-w3g128-calpt   rtn-w3g128-onlymamba

Uso: python lumi/scripts/quantize.py --base $LUMI gptq-w4g128-calen gptq-w4g128-calpt
Saida: $LUMI_CKPTS/<base_curto>__<spec>/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.environ["GLOD_ROOT"])

CKPTS = Path(os.environ["LUMI_CKPTS"])
DATA = Path(os.environ["LUMI_DATA"])
SPEC_RE = re.compile(r"(rtn|gptq|awq)-w(\d)g(\d+)(?:-cal(en|pt))?(?:-(only|skip)(mamba|mlp|attn))?(?:-s(\d+))?")
BLOCK_RE = {"mamba": r"\.mixer\.(in|out)_proj$", "mlp": r"\.mixer\.(up|down)_proj$",
            "attn": r"\.mixer\.[qkvo]_proj$"}


def short(base: str) -> str:
    return {"ceia-research/lumi-4b": "lumi4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": "nemo4b"}.get(
        base, base.split("/")[-1].lower())


def ckpt_dir(base: str, spec: str) -> Path:
    return CKPTS / f"{short(base)}__{spec}"


def calibration(tok, lang: str, n_seq: int = 128, seq_len: int = 512, seed: int = 0) -> torch.Tensor:
    """n_seq janelas de seq_len tokens, um documento por janela (mesma amostragem do GLOD)."""
    texts = pd.read_parquet(DATA / f"web_{lang}.parquet")["text"].tolist()
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in torch.randperm(len(texts), generator=g).tolist():
        ids = tok(texts[i], return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(ids) > seq_len:
            s = int(torch.randint(0, len(ids) - seq_len, (1,), generator=g))
            rows.append(ids[s:s + seq_len])
        if len(rows) == n_seq:
            break
    batch = torch.stack(rows)
    if tok.bos_token_id is not None:
        batch = torch.cat([torch.full((n_seq, 1), tok.bos_token_id), batch[:, :-1]], dim=1)
    return batch


def apply(model, tok, spec: str) -> dict:
    from glod.core.compressors import WeightBank, calibrated_compress, rtn

    m = SPEC_RE.fullmatch(spec)
    if not m:
        raise ValueError(spec)
    method, bits, group, cal = m[1], int(m[2]), int(m[3]), m[4] or "en"
    only = m[6] if m[5] == "only" else None
    skip = m[6] if m[5] == "skip" else None
    seed = int(m[7]) if m[7] else 0
    bank = WeightBank(model.model)
    n_all = len(bank.targets)
    if only:
        bank.targets = [t for t in bank.targets if re.search(BLOCK_RE[only], t.name)]
    if skip:   # precisao mista: o bloco fica em BF16
        bank.targets = [t for t in bank.targets if not re.search(BLOCK_RE[skip], t.name)]
    info = {"method": method, "bits": bits, "group": group, "only": only, "skip": skip,
            "n_targets": len(bank.targets), "n_linear_total": n_all,
            "params_quantized": bank.total_params()}
    if method == "rtn":
        bank.map(lambda w, t: rtn(w, bits, group))
    else:
        calib = calibration(tok, cal, seed=seed).to(model.device)
        info["calib"] = cal
        info["seed"] = seed
        out = calibrated_compress(model, bank, model.model.layers, calib, method, bits=bits, group=group,
                                  batch_size=1)   # o scan do Mamba2 em PyTorch puro gasta ~6 GB por sequencia
        info.update({k: v for k, v in out.items() if k != "alphas"})
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("specs", nargs="+")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    for spec in args.specs:
        out = ckpt_dir(args.base, spec)
        if (out / "lumi_meta.json").exists():
            print(f"[skip] {out}", flush=True)
            continue
        t0 = time.time()
        # recarrega a cada spec: o WeightBank filtrado nao restaura o que ficou fora dele
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda").eval()
        info = apply(model, tok, spec)
        out.mkdir(parents=True, exist_ok=True)
        model.generation_config.do_sample = True   # Nemotron: top_p 0.95 sem do_sample nao salva no transformers 5
        model.save_pretrained(out)
        tok.save_pretrained(out)
        meta = {"base": args.base, "spec": spec, "info": info, "seconds": round(time.time() - t0, 1)}
        (out / "lumi_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[ok] {out} ({meta['seconds']} s) {info}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
