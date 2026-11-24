"""Tabelas e testes pre-registrados do E1 (e qualidade do E2) a partir de results/e1/{eval,tv}.

Bootstrap pareado por item (10 000 reamostragens, por tarefa), IC 95% percentil.
Saidas em results/e1/: acc.csv, tests.csv, tv.csv, blocks.csv, summary.json
Uso: python lumi/scripts/analyze.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(os.environ["LUMI_RESULTS"]) / "e1"
B = 10_000
RNG = np.random.default_rng(0)
PT_TASKS = ["mmlu_pt", "enem", "bluex", "oab"]
SPECS = ["rtn-w4g128", "gptq-w4g128-calen", "gptq-w4g128-calpt", "rtn-w3g128", "gptq-w3g128-calen",
         "gptq-w3g128-calpt", "rtn-w3g128-onlymamba", "rtn-w3g128-onlymlp", "rtn-w3g128-onlyattn"]
W3 = ["rtn-w3g128", "gptq-w3g128-calen", "gptq-w3g128-calpt"]


def load_eval() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in glob.glob(str(R / "eval" / "*.parquet"))], ignore_index=True)
    d["base"] = d["tag"].str.split("__").str[0]
    d["spec"] = d["tag"].str.split("__").str[1]
    return d


def matrix(d: pd.DataFrame, task: str, tags: list[str]) -> np.ndarray | None:
    """Corretude [itens x tags] nos itens comuns a todas as tags."""
    sub = d[(d.task == task) & d.tag.isin(tags)]
    if sub.tag.nunique() < len(tags):
        return None
    m = sub.pivot_table(index="id", columns="tag", values="correct", aggfunc="first").dropna()
    return m[tags].to_numpy(dtype=float)


def boot(stat, mats: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """stat(dict task -> matriz) -> escalar; reamostra itens dentro de cada tarefa."""
    point = stat(mats)
    vals = np.empty(B)
    idx = {t: RNG.integers(0, len(m), size=(B, len(m))) for t, m in mats.items()}
    for b in range(B):
        vals[b] = stat({t: m[idx[t][b]] for t, m in mats.items()})
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def pooled(mats: dict[str, np.ndarray], col: int) -> float:
    n = sum(len(m) for m in mats.values())
    return sum(m[:, col].sum() for m in mats.values()) / n


def tests(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SPECS:
        # H2: dentro do Lumi, queda em PT - queda em EN nos itens pareados do MMLU
        pt = matrix(d, "mmlu_pt", ["lumi4b__bf16", f"lumi4b__{spec}"])
        en = matrix(d, "mmlu_en", ["lumi4b__bf16", f"lumi4b__{spec}"])
        if pt is not None and en is not None:
            m = {"x": np.concatenate([pt, en], axis=1)}       # mesma ordem de id nos dois idiomas
            f = lambda M: ((M["x"][:, 0] - M["x"][:, 1]) - (M["x"][:, 2] - M["x"][:, 3])).mean()
            rows.append(("H2", spec, "queda PT - queda EN (Lumi, MMLU pareado)", *boot(f, m)))
        # H1: queda do ganho Lumi - Nemotron em PT (MMMLU + provas) menos a queda do ganho em EN
        tags = ["lumi4b__bf16", f"lumi4b__{spec}", "nemo4b__bf16", f"nemo4b__{spec}"]
        mp = {t: matrix(d, t, tags) for t in PT_TASKS}
        me = matrix(d, "mmlu_en", tags)
        if me is not None and all(v is not None for v in mp.values()):
            def drop_gain(M):
                g0 = pooled(M, 0) - pooled(M, 2)
                g1 = pooled(M, 1) - pooled(M, 3)
                return g0 - g1
            allm = {**{f"pt_{k}": v for k, v in mp.items()}, "en": me}
            f = lambda M: drop_gain({k: v for k, v in M.items() if k.startswith("pt_")}) - drop_gain({"en": M["en"]})
            rows.append(("H1", spec, "queda do ganho Lumi-Nemotron: PT - EN", *boot(f, allm)))
            f_pt = lambda M: drop_gain({k: v for k, v in M.items() if k.startswith("pt_")})
            rows.append(("H1a", spec, "queda do ganho Lumi-Nemotron em PT", *boot(f_pt, allm)))
            f_en = lambda M: drop_gain({"en": M["en"]})
            rows.append(("H1b", spec, "queda do ganho Lumi-Nemotron em EN", *boot(f_en, allm)))
    # H3: GPTQ cal-pt vs cal-en, queda em PT no Lumi (MMMLU + provas)
    for w in ("w3", "w4"):
        tags = ["lumi4b__bf16", f"lumi4b__gptq-{w}g128-calen", f"lumi4b__gptq-{w}g128-calpt"]
        mp = {t: matrix(d, t, tags) for t in PT_TASKS}
        if all(v is not None for v in mp.values()):
            f = lambda M: pooled(M, 2) - pooled(M, 1)          # ganho de acuracia da calibracao PT
            rows.append(("H3", f"gptq-{w}", "acc PT: cal-pt - cal-en (Lumi)", *boot(f, mp)))
            me = {"en": matrix(d, "mmlu_en", tags)}
            if me["en"] is not None:
                rows.append(("H3b", f"gptq-{w}", "acc EN: cal-pt - cal-en (Lumi)", *boot(f, me)))
    return pd.DataFrame(rows, columns=["teste", "config", "medida", "valor", "ic_lo", "ic_hi"])


def tv_table() -> pd.DataFrame:
    fs = glob.glob(str(R / "tv" / "*.parquet"))
    if not fs:
        return pd.DataFrame()
    t = pd.concat([pd.read_parquet(f) for f in fs])
    w = t.groupby(["tag", "corpus"]).apply(lambda g: pd.Series(
        {k: np.average(g[k], weights=g.n_tok) for k in ("tv", "kl", "flip")}), include_groups=False)
    return w.unstack("corpus")


def main() -> None:
    d = load_eval()
    acc = d.pivot_table(index="tag", columns="task", values="correct", aggfunc="mean")
    pt_items = d[d.task.isin(PT_TASKS)].groupby("tag").correct.mean()
    acc["pt_pool"] = pt_items
    acc.round(4).to_csv(R / "acc.csv")
    print(acc.round(3).to_string())
    te = tests(d)
    te.round(4).to_csv(R / "tests.csv", index=False)
    print(te.round(4).to_string())
    tv = tv_table()
    summary = {"acc": acc.round(4).reset_index().to_dict("records"), "tests": te.round(4).to_dict("records")}
    if len(tv):
        tv.round(5).to_csv(R / "tv.csv")
        print(tv.round(4).to_string())
        # H4: Spearman entre TV no texto PT e a queda de acuracia PT, por modelo
        h4 = []
        for base in ("lumi4b", "nemo4b"):
            rows = []
            for spec in SPECS:
                tag = f"{base}__{spec}"
                if tag in tv.index and tag in acc.index and f"{base}__bf16" in acc.index:
                    for lang, corp, col in (("pt", "web_pt", "pt_pool"), ("en", "web_en", "mmlu_en")):
                        rows.append((lang, tv.loc[tag, ("tv", corp)], acc.loc[f"{base}__bf16", col] - acc.loc[tag, col]))
            for lang in ("pt", "en"):
                xs = [(a, b) for l, a, b in rows if l == lang]
                if len(xs) >= 4:
                    rho = spearmanr([a for a, _ in xs], [b for _, b in xs]).statistic
                    h4.append({"base": base, "lang": lang, "n": len(xs), "spearman": round(float(rho), 3)})
        print(pd.DataFrame(h4).to_string())
        summary["h4"] = h4
        summary["tv"] = json.loads(tv.round(5).to_json(orient="index"))
    (R / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
