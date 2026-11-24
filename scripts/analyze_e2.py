"""Tabelas do E2 (velocidade no vLLM, GGUF no llama.cpp) a partir de results/e2/.

vLLM (protocolo da sparkbench: 512 -> 256, ignore_eos, carga fechada c = 1..128):
  saida por nivel: tok/s de saida, TTFT/TPOT P50/P99, goodput SLO B (TTFT <= 1 s, TPOT <= 50 ms),
  potencia media (nvidia-smi a cada 200 ms, so amostras com GPU ocupada) e tokens de saida por joule.
  teto de banda com 1 usuario: 3,35 TB/s / bytes lidos por passo (pesos + estado), fracao atingida.
llama.cpp: KLD/top-1/PPL contra o BF16 (llama-perplexity) e pp512/tg256 (llama-bench).
Saidas em results/e2/: serve.csv, serve_summary.csv, gguf.csv
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(os.environ["LUMI_RESULTS"]) / "e2"
BW = 3.35e12                      # H100 SXM, HBM3
N_LIN, N_ALL = 3.15e9, 3.97e9      # parametros nas Linear do decoder / total (Lumi e Nemotron, mesma arquitetura)
# bytes de pesos lidos por passo de decode (embeddings nao sao lidos inteiros; lm_head e)
EMB = 131072 * 3136
WBYTES = {
    "bf16": 2 * (N_ALL - EMB), "nemo-bf16": 2 * (N_ALL - EMB),
    "fp8": 1 * N_LIN + 2 * (N_ALL - N_LIN - EMB), "fp8kv": 1 * N_LIN + 2 * (N_ALL - N_LIN - EMB),
    "w4a16": N_LIN * (0.5 + 2 / 64) + 2 * (N_ALL - N_LIN - EMB),      # int4 + escala bf16 por grupo de 64
}
N_ATTN = 4 * (3136 * 4096 + 2 * 3136 * 1024 + 4096 * 3136)
WBYTES["w4a16-skipattn"] = WBYTES["w4a16"] + N_ATTN * (2 - 0.5 - 2 / 64)    # atencao em BF16
STATE = 41e6 + 16384 * 640        # estado Mamba + KV medio (512 + 128 tokens) por sequencia
NAMES = {  # nemo-bf16-topp095: primeira medicao, top_p 0.95 do generation_config (apendice)
"bf16": "Lumi BF16", "fp8": "Lumi FP8 (W8A8)", "fp8kv": "Lumi FP8 + KV FP8",
         "w4a16": "Lumi W4A16 (cal. PT)", "w4a16-skipattn": "Lumi W4A16 + atenção BF16", "nemo-bf16": "Nemotron base BF16"}


def serve() -> pd.DataFrame:
    rows = []
    for v in NAMES:
        if not (R / "bench" / v).exists():
            continue
        for f in sorted((R / "bench" / v).glob("c*.json")):
            d = json.loads(f.read_text())
            c = int(f.stem[1:])
            p = pd.read_csv(R / "bench" / v / f"power_c{c}.csv", header=None, skipinitialspace=True,
                            names=["ts", "w", "util", "sm", "mem"])
            p["w"] = pd.to_numeric(p["w"].astype(str).str.replace("W", ""), errors="coerce")
            busy = p[pd.to_numeric(p["util"].astype(str).str.replace("%", ""), errors="coerce") > 0]
            watts = float(busy["w"].mean()) if len(busy) else float(p["w"].mean())
            rows.append({
                "variante": v, "c": c, "tok_s": d["output_throughput"],
                "ttft_p50": d["p50_ttft_ms"], "ttft_p99": d["p99_ttft_ms"],
                "tpot_p50": d["p50_tpot_ms"], "tpot_p99": d["p99_tpot_ms"],
                "goodput_frac": d["request_goodput"] / d["request_throughput"],
                "req_s": d["request_throughput"], "watts": watts, "tok_J": d["output_throughput"] / watts,
            })
    return pd.DataFrame(rows).sort_values(["variante", "c"])


def summary(s: pd.DataFrame) -> pd.DataFrame:
    out = []
    for v, g in s.groupby("variante", sort=False):
        g = g.set_index("c")
        ok = g[g.goodput_frac >= 0.99]
        ceil = BW / (WBYTES[v] + STATE)
        tok1 = 1000 / g.loc[1, "tpot_p50"]
        out.append({
            "variante": NAMES[v], "tok_s_c1": tok1, "teto_c1": ceil, "frac_teto": tok1 / ceil,
            "ttft_p50_c1": g.loc[1, "ttft_p50"], "ttft_p99_c16": g.loc[16, "ttft_p99"],
            "tpot_p50_c64": g.loc[64, "tpot_p50"], "tok_s_max": g.tok_s.max(), "c_max": int(g.tok_s.idxmax()),
            "slo_b_max_c": int(ok.index.max()) if len(ok) else 0, "tok_J_c1": g.loc[1, "tok_J"],
            "tok_J_max": g.tok_J.max(), "watts_c128": g.loc[128, "watts"],
            "pesos_GB": WBYTES[v] / 1e9,
        })
    return pd.DataFrame(out)


def gguf() -> pd.DataFrame:
    G = R / "gguf"
    rows = []
    for V in ("q8_0", "q4km", "q4km-imaten", "q4km-imatpt"):
        r = {"variante": V}
        for L in ("pt", "en"):
            f = G / f"kld_{V}_{L}.log"
            if not f.exists():
                continue
            t = f.read_text(errors="replace")
            for key, pat in (("kld", r"Mean    KLD:\s+([\d.]+)"), ("top1", r"Same top p:\s+([\d.]+)"),
                             ("ppl_ratio", r"Mean PPL\(Q\)/PPL\(base\)\s+:\s+([\d.]+)"),
                             ("kld_p99", r"99\.0%   KLD:\s+([\d.]+)")):
                m = re.search(pat, t)
                r[f"{key}_{L}"] = float(m.group(1)) if m else np.nan
        q = G / f"quantize_{V}.log"
        if q.exists():
            t = q.read_text(errors="replace")
            m = re.search(r"model size\s+=\s+([\d.]+) MiB", t.split("quant size")[0][-400:] + t)
            m2 = re.search(r"quant size\s+=\s+([\d.]+) MiB", t)
            r["MiB"] = float(m2.group(1)) if m2 else np.nan
            r["fallbacks"] = len(re.findall(r"fallback", t, flags=re.I))
        b = G / f"bench_{V}.json"
        rows.append(r)
    for V in ("bf16", "q8_0", "q4km", "q4km-imaten", "q4km-imatpt"):
        b = G / f"bench_{V}.json"
        if b.exists() and b.stat().st_size:
            for e in json.loads(b.read_text()):
                k = "pp512_tok_s" if e["n_prompt"] else "tg256_tok_s"
                row = next((x for x in rows if x["variante"] == V), None)
                if row is None:
                    row = {"variante": V}; rows.append(row)
                row[k] = e["avg_ts"]; row[k.replace("tok_s", "sd")] = e["stddev_ts"]
    return pd.DataFrame(rows)


def main() -> None:
    s = serve()
    s.round(4).to_csv(R / "serve.csv", index=False)
    sm = summary(s)
    sm.round(4).to_csv(R / "serve_summary.csv", index=False)
    print(sm.round(3).to_string())
    g = gguf()
    g.round(5).to_csv(R / "gguf.csv", index=False)
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()


def quality_tests() -> pd.DataFrame:
    """Diferencas pareadas de acuracia (pool PT e MMLU EN) com bootstrap por item, IC 95%."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze import PT_TASKS, boot, load_eval, matrix, pooled
    d = load_eval()
    pairs = [("real-fp8", "bf16"), ("real-fp8kv", "bf16"), ("real-w4a16-calpt", "bf16"),
             ("real-w4a16-calen", "bf16"), ("real-w4a16-calpt", "real-w4a16-calen"),
             ("gptq-w4g64-calpt", "real-w4a16-calpt"),
             ("real-w4a16-calpt-skipattn", "bf16"), ("real-w4a16-calpt-skipattn", "real-w4a16-calpt"),
             ("gptq-w3g128-calpt-skipattn", "gptq-w3g128-calpt"), ("gptq-w3g128-calen-skipattn", "gptq-w3g128-calen"),
             ("rtn-w4g128-skipattn", "rtn-w4g128"), ("rtn-w3g128-skipattn", "rtn-w3g128"),
             ("gptq-w3g128-calpt-skipattn", "bf16"), ("rtn-w4g128-skipattn", "bf16"),
             ("gguf-q8_0", "gguf-bf16"), ("gguf-q4km", "gguf-bf16"), ("gguf-q4km-imatpt", "gguf-bf16"),
             ("gguf-q4km-imatpt", "gguf-q4km"), ("gguf-q4km-imaten", "gguf-q4km"),
             ("gguf-q4km-imatpt", "gguf-q4km-imaten"), ("gguf-bf16", "bf16")]
    rows = []
    for a, b in pairs:
        tags = [f"lumi4b__{b}", f"lumi4b__{a}"]
        mp = {t: matrix(d, t, tags) for t in PT_TASKS}
        me = {"en": matrix(d, "mmlu_en", tags)}
        f = lambda M: pooled(M, 1) - pooled(M, 0)
        if all(v is not None for v in mp.values()):
            rows.append((a, b, "pt", sum(len(v) for v in mp.values()), *boot(f, mp)))
        if me["en"] is not None:
            rows.append((a, b, "en", len(me["en"]), *boot(f, me)))
    return pd.DataFrame(rows, columns=["a", "b", "idioma", "n", "a_menos_b", "ic_lo", "ic_hi"])


if __name__ == "__main__" and os.environ.get("E2_TESTS", "1") == "1":
    q = quality_tests()
    q.round(4).to_csv(R / "quality_tests.csv", index=False)
    print(q.round(4).to_string())


def e4_tests() -> pd.DataFrame:
    """E4: Nemotron + mudanca do Lumi truncada no posto r, contra a Nemotron (pool PT, bootstrap por item)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze import PT_TASKS, boot, load_eval, matrix, pooled
    d = load_eval()
    rows = []
    for a, b in [(f"nemo4b__plus-delta-r{r}", "nemo4b__bf16") for r in (0, 16, 64, 256)] + \
                [("lumi4b__bf16", "nemo4b__bf16"), ("lumi4b__bf16", "nemo4b__plus-delta-r256")]:
        ts = [b, a]
        mp = {t: matrix(d, t, ts) for t in PT_TASKS}
        if all(v is not None for v in mp.values()):
            rows.append((a, b, *boot(lambda M: pooled(M, 1) - pooled(M, 0), mp)))
    return pd.DataFrame(rows, columns=["a", "b", "a_menos_b", "ic_lo", "ic_hi"])


if __name__ == "__main__" and os.environ.get("E4_TESTS", "1") == "1":
    (R.parent / "e4").mkdir(exist_ok=True)
    e4 = e4_tests()
    e4.round(5).to_csv(R.parent / "e4" / "tests.csv", index=False)
    print(e4.round(4).to_string())
