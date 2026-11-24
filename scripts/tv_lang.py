"""Divergencia por token (estilo GLOD) entre o pai BF16 e um comprimido, separada por idioma.

Teacher forcing nos dois modelos sobre os MESMOS textos; por posicao:
    tv   = 0.5 * sum_v |p(v) - q(v)|     (softmax cru, T=1)
    kl   = KL(p || q)
    flip = argmax p != argmax q
Corpora (n textos cada, ate L tokens):
    web_pt / web_en     FineWeb-2 por_Latn / C4 en (texto corrido)
    mmlu_pt / mmlu_en   os mesmos itens do MMLU nos dois idiomas, no formato de chat da avaliacao
Saida: $LUMI_RESULTS/e1/tv/<tag>.parquet (uma linha por texto: medias e n de tokens)

Uso: python lumi/scripts/tv_lang.py --parent $LUMI --comp $LUMI_CKPTS/lumi4b__rtn-w4g128 --tag lumi4b__rtn-w4g128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_mc import load_task, user_msg  # noqa: E402

DATA = Path(os.environ["LUMI_DATA"])
OUT = Path(os.environ["LUMI_RESULTS"]) / "e1" / "tv"


def corpora(tok, n: int, L: int) -> dict[str, list[torch.Tensor]]:
    out = {}
    for lang in ("pt", "en"):
        texts = pd.read_parquet(DATA / f"web_{lang}.parquet")["text"].tolist()
        ids = []
        for t in texts[1000:]:            # os primeiros 1000 nao entram (a calibracao sorteia do arquivo todo,
            x = tok(t, add_special_tokens=False).input_ids   # mas so 128 janelas: sobreposicao desprezivel)
            if len(x) >= L:
                ids.append(torch.tensor([tok.bos_token_id] + x[:L - 1]))
            if len(ids) == n:
                break
        out[f"web_{lang}"] = ids
        df = load_task(f"mmlu_{lang}").sample(n=n, random_state=0)   # mesmo sorteio nos dois idiomas
        out[f"mmlu_{lang}"] = [torch.tensor(tok.apply_chat_template(
            [{"role": "user", "content": user_msg(r)}], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False)[:L]) for r in df.itertuples()]
    return out


@torch.no_grad()
def logits(model, batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    T = max(len(x) for x in batch)
    ids = torch.zeros(len(batch), T, dtype=torch.long)
    mask = torch.zeros(len(batch), T, dtype=torch.bool)
    for i, x in enumerate(batch):           # padding a direita: o hibrido e causal, o pad nao afeta o prefixo
        ids[i, :len(x)] = x
        mask[i, :len(x)] = True
    out = model(input_ids=ids.cuda(), use_cache=False).logits.float()
    return out, mask.cuda()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--comp", nargs="+", required=True, help="um ou mais checkpoints comprimidos")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    assert len(args.comp) == len(args.tags)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [(c, t) for c, t in zip(args.comp, args.tags) if not (OUT / f"{t}.parquet").exists()]
    if not todo:
        print("[skip] tudo pronto")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.parent)
    data = corpora(tok, args.n, args.L)
    P = AutoModelForCausalLM.from_pretrained(args.parent, dtype=torch.bfloat16).cuda().eval()
    # log-probs do pai calculados uma vez por lote e reaproveitados para todos os comprimidos
    for comp, tag in todo:
        t0 = time.time()
        Q = AutoModelForCausalLM.from_pretrained(comp, dtype=torch.bfloat16).cuda().eval()
        rows = []
        for corpus, seqs in data.items():
            for s in range(0, len(seqs), args.bs):
                batch = seqs[s:s + args.bs]
                lp, mask = logits(P, batch)
                lq, _ = logits(Q, batch)
                lp, lq = lp.log_softmax(-1), lq.log_softmax(-1)
                p = lp.exp()
                tv = 0.5 * (p - lq.exp()).abs().sum(-1)
                kl = (p * (lp - lq)).sum(-1)
                flip = lp.argmax(-1) != lq.argmax(-1)
                for i in range(len(batch)):
                    m = mask[i]
                    rows.append({"corpus": corpus, "idx": s + i, "n_tok": int(m.sum()),
                                 "tv": tv[i][m].mean().item(), "kl": kl[i][m].mean().item(),
                                 "flip": flip[i][m].float().mean().item()})
                del lp, lq, p, tv, kl
        df = pd.DataFrame(rows)
        df["tag"], df["comp"] = tag, comp
        df.to_parquet(OUT / f"{tag}.parquet")
        print(df.groupby("corpus")[["tv", "kl", "flip"]].mean().round(4).to_string())
        print(f"[ok] {tag} ({time.time() - t0:.0f} s)", flush=True)
        del Q
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
