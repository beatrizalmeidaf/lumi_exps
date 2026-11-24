"""E4: o que o treino em PT mudou nos pesos (Lumi - Nemotron) e onde essa mudanca age.

Para cada Linear do decoder:
  * rel  = ||dW|| / ||W_pai||                      (tamanho da mudanca)
  * r_L  = sqrt(tr(dW H_L dW^T) / tr(W H_L W^T))   (mudanca relativa da SAIDA com entradas do idioma L,
           H_L = E[x x^T] das entradas da camada em texto web L, medido no Lumi)
  * e_k  = fracao de ||dW||^2 nos k maiores valores singulares (k = 16, 64, 256)
Se r_pt >> r_en, a mudanca do CPT age sobretudo em entradas em portugues: e o que uma calibracao em ingles
(que pondera os pesos por H_en) trata como pouco importante.

Com --ranks, tambem salva Nemotron + SVD-truncamento de dW no posto r (um "adaptador" de posto r) para avaliar.
Saidas: results/e4/delta_layers.csv; checkpoints $LUMI_CKPTS/nemo4b__plus-delta-r<r>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path(os.environ["LUMI_DATA"])
OUT = Path(os.environ["LUMI_RESULTS"]) / "e4"
CK = Path(os.environ["LUMI_CKPTS"])
LIN = re.compile(r"layers\.(\d+)\.mixer\.(in_proj|out_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)$")
KIND = {"in_proj": "mamba", "out_proj": "mamba", "up_proj": "mlp", "down_proj": "mlp",
        "q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn"}


def second_moments(model, tok, lang: str, n: int, L: int) -> dict[str, torch.Tensor]:
    texts = pd.read_parquet(DATA / f"web_{lang}.parquet")["text"].tolist()[2000:]   # fora de calibracao e TV
    H, cnt, hooks = {}, {}, []
    for name, mod in model.named_modules():
        if LIN.search(name):
            def hk(m, inp, out, name=name):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                H[name] = H.get(name, 0) + x.T @ x
                cnt[name] = cnt.get(name, 0) + x.shape[0]
            hooks.append(mod.register_forward_hook(hk))
    k = 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, add_special_tokens=False).input_ids
            if len(ids) < L:
                continue
            model(torch.tensor([[tok.bos_token_id] + ids[:L - 1]], device="cuda"))
            k += 1
            if k == n:
                break
    for h in hooks:
        h.remove()
    return {k_: v / cnt[k_] for k_, v in H.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--ranks", type=int, nargs="*", default=[])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lumi_id, nemo_id = os.environ["LUMI"], os.environ["NEMO"]
    tok = AutoTokenizer.from_pretrained(lumi_id)
    lumi = AutoModelForCausalLM.from_pretrained(lumi_id, dtype=torch.bfloat16).to("cuda").eval()
    nemo_sd = AutoModelForCausalLM.from_pretrained(nemo_id, dtype=torch.bfloat16).state_dict()

    csv = OUT / "delta_layers.csv"
    if not csv.exists():
        Hs = {lang: second_moments(lumi, tok, lang, a.n, a.L) for lang in ("pt", "en")}
        rows = []
        for name, mod in lumi.named_modules():
            m = LIN.search(name)
            if not m:
                continue
            W1 = mod.weight.float()
            W0 = nemo_sd[name + ".weight"].to("cuda").float()
            D = W1 - W0
            r = {"layer": int(m[1]), "proj": m[2], "kind": KIND[m[2]], "rel": (D.norm() / W0.norm()).item()}
            for lang, H in Hs.items():
                h = H[name]
                r[f"r_{lang}"] = ((D @ h * D).sum() / (W0 @ h * W0).sum()).clamp_min(0).sqrt().item()
            s2 = torch.linalg.svdvals(D) ** 2
            for k in (16, 64, 256):
                r[f"e{k}"] = (s2[:k].sum() / s2.sum()).item()
            r["n_params"] = D.numel()
            rows.append(r)
            del W0, D
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)
    df = pd.read_csv(csv)
    df["ratio"] = df.r_pt / df.r_en
    w = lambda g, c: (g[c] * g.n_params).sum() / g.n_params.sum()
    print(df.groupby("kind").apply(lambda g: pd.Series({c: w(g, c) for c in ("rel", "r_pt", "r_en", "ratio", "e16", "e64", "e256")}),
                                   include_groups=False).round(4).to_string())
    print("todas:", {c: round(w(df, c), 4) for c in ("rel", "r_pt", "r_en", "ratio", "e64", "e256")})

    # Nemotron + dW truncado no posto r, para avaliar quanto do ganho em PT um adaptador de posto r carrega
    for rk in a.ranks:
        out = CK / f"nemo4b__plus-delta-r{rk}"
        if (out / "lumi_meta.json").exists():
            continue
        with torch.no_grad():
            for name, mod in lumi.named_modules():
                if not LIN.search(name):
                    continue
                W1 = mod.weight.float()
                W0 = nemo_sd[name + ".weight"].to("cuda").float()
                U, S, Vh = torch.linalg.svd(W1 - W0, full_matrices=False)
                mod.weight.copy_((W0 + (U[:, :rk] * S[:rk]) @ Vh[:rk]).to(torch.bfloat16))
            # o resto (embeddings, normas, conv, A/D do Mamba, lm_head) fica do Lumi; so as Linear viram pai + posto r
            lumi.generation_config.do_sample = True
            lumi.save_pretrained(out)
            tok.save_pretrained(out)
            n_ad = sum((W.shape[0] + W.shape[1]) * rk for n_, W in
                       ((n_, m_.weight) for n_, m_ in lumi.named_modules() if LIN.search(n_)))
            (out / "lumi_meta.json").write_text(json.dumps({"rank": rk, "adapter_params": n_ad}))
            print(f"[ok] {out} adaptador {n_ad / 1e6:.1f} M params")
        # recarrega o Lumi para o proximo posto
        lumi = AutoModelForCausalLM.from_pretrained(lumi_id, dtype=torch.bfloat16).to("cuda").eval()


if __name__ == "__main__":
    main()
