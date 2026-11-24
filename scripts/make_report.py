"""Relatorio E1 + E2 do Lumi-4B (HTML -> PDF), no formato dos relatorios da sparkbench.

Le results/e1/{acc,tests,tv}.csv e results/e2/{serve,serve_summary,gguf,quality_tests,acc_common_items}.csv,
gera as figuras (SVG embutido) e escreve relatorios/lumi4b_compressao.{html,pdf}.
O texto interpretativo fica em relatorios/texto.py (dicionario TXT), separado dos numeros.
Uso: python lumi/scripts/make_report.py   (venv com matplotlib e playwright)
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "relatorios"

# paleta categorica (ordem fixa) + neutro para a referencia
C = {"bf16": "#2a78d6", "fp8": "#eb6834", "w4a16": "#1baf7a", "fp8kv": "#eda100", "nemo-bf16": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "lines.linewidth": 2,
})
LABEL = {"bf16": "Lumi BF16", "fp8": "Lumi FP8", "fp8kv": "Lumi FP8 + KV FP8", "w4a16": "Lumi W4A16",
         "nemo-bf16": "Nemotron BF16"}


def svg(fig) -> str:
    b = io.StringIO()
    fig.savefig(b, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = b.getvalue()
    return s[s.index("<svg"):]


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}".replace(".", ",")


def num(x, d=0) -> str:
    s = f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def pp(x, d=1) -> str:
    s = f"{100 * x:+.{d}f}".replace(".", ",").replace("-", "−")
    return s


def ci(lo, hi, d=1) -> str:
    return f"[{pp(lo, d)}; {pp(hi, d)}]"


def table(df: pd.DataFrame, cls: str = "") -> str:
    h = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df.astype(str).values)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table>'


def leitura(items: list[str]) -> str:
    li = "".join(f"<li>{i}</li>" for i in items)
    return f'<div class="leitura"><div class="lt">LEITURA</div><ul>{li}</ul></div>'


def kpi(tiles: list[tuple[str, str, str]]) -> str:
    t = "".join(f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{l}</div><div class="ks">{s}</div></div>'
                for v, l, s in tiles)
    return f'<div class="kpis">{t}</div>'


# ---------------------------------------------------------------- figuras E2
def fig_throughput(s: pd.DataFrame) -> str:
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 2.9))
    for v in ("bf16", "fp8", "w4a16", "nemo-bf16"):
        g = s[s.variante == v].sort_values("c")
        ls = "--" if v == "nemo-bf16" else "-"
        axs[0].plot(g.c, g.tok_s, ls, color=C[v], marker="o", ms=3.5, label=LABEL[v])
        axs[1].plot(g.c, g.tpot_p50, ls, color=C[v], marker="o", ms=3.5)
        axs[2].plot(g.c, g.ttft_p99, ls, color=C[v], marker="o", ms=3.5)
    for a, t in zip(axs, ("tokens de saída/s (agregado)", "TPOT P50 (ms)", "TTFT P99 (ms)")):
        a.set_xscale("log", base=2)
        a.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
        a.set_xlabel("usuários simultâneos")
        a.set_title(t, fontsize=9, color=INK, loc="left")
    axs[2].axhline(1000, color="#e34948", lw=1, ls=":")
    axs[2].text(1, 1030, "SLO B: 1 s", color=INK2, fontsize=7.5)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return svg(fig)


def fig_energy(s: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for v in ("bf16", "fp8", "w4a16", "nemo-bf16"):
        g = s[s.variante == v].sort_values("c")
        ax.plot(g.c, g.tok_J, "--" if v == "nemo-bf16" else "-", color=C[v], marker="o", ms=3.5, label=LABEL[v])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 128], ["1", "4", "16", "64", "128"])
    ax.set_xlabel("usuários simultâneos")
    ax.set_title("tokens de saída por joule (GPU)", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return svg(fig)


def fig_frontier(points: list[tuple[str, float, float, str]]) -> str:
    """(rotulo, GB de pesos, acc PT, chave de cor)."""
    fig, ax = plt.subplots(figsize=(10.2, 3.0))
    off = {"GGUF Q8_0": (6, 4), "FP8": (6, -10), "GGUF Q4_K_M": (6, -12), "Q4_K_M imat. PT": (6, 3),
           "Q4_K_M imat. EN": (6, 4), "W4A16 cal. PT": (-20, -13), "W4A16 cal. EN": (6, -3), "BF16": (-28, 4)}
    for lab, x, y, k in points:
        ax.scatter([x], [100 * y], s=48, color=C.get(k, INK2), edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(lab, (x, 100 * y), xytext=off.get(lab, (6, 3)), textcoords="offset points", fontsize=7.5,
                    color=INK2)
    ax.set_xlabel("GB de pesos lidos por passo")
    ax.set_ylabel("acurácia PT (%)")
    ax.set_title("qualidade × tamanho (itens comuns)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return svg(fig)


# ---------------------------------------------------------------- figuras E1
def fig_drops(acc: pd.DataFrame, specs: list[str], names: dict[str, str]) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.0), sharey=True)
    for a, base, tit in zip(axs, ("lumi4b", "nemo4b"), ("Lumi-4B", "Nemotron-3-Nano-4B base")):
        if f"{base}__bf16" not in acc.index:
            a.set_visible(False)
            continue
        ys = np.arange(len(specs))
        b = acc.loc[f"{base}__bf16"]
        dpt = [b.pt_pool - acc.loc[f"{base}__{s}", "pt_pool"] if f"{base}__{s}" in acc.index else np.nan for s in specs]
        den = [b.mmlu_en - acc.loc[f"{base}__{s}", "mmlu_en"] if f"{base}__{s}" in acc.index else np.nan for s in specs]
        a.barh(ys - 0.19, 100 * np.array(dpt), 0.36, color="#2a78d6", label="PT (MMMLU + provas)")
        a.barh(ys + 0.19, 100 * np.array(den), 0.36, color="#eb6834", label="EN (MMLU)")
        a.set_yticks(ys, [names[s] for s in specs])
        a.invert_yaxis()
        a.set_xlabel("queda de acurácia vs BF16 (pp)")
        a.set_title(tit, fontsize=9, color=INK, loc="left")
        a.axvline(0, color=INK2, lw=0.8)
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return svg(fig)


def fig_tv(tv: pd.DataFrame, acc: pd.DataFrame, specs: list[str]) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.0))
    for a, lang, corp, col in ((axs[0], "PT", "web_pt", "pt_pool"), (axs[1], "EN", "web_en", "mmlu_en")):
        for base, k, lab in (("lumi4b", "bf16", "Lumi"), ("nemo4b", "nemo-bf16", "Nemotron")):
            xs, ys = [], []
            for s in specs:
                t = f"{base}__{s}"
                if t in tv.index and t in acc.index:
                    xs.append(tv.loc[t, f"tv_{corp}"])
                    ys.append(100 * (acc.loc[f"{base}__bf16", col] - acc.loc[t, col]))
            a.scatter(xs, ys, s=48, color=C[k], edgecolor="white", linewidth=1.5, label=lab, zorder=3)
        a.set_xlabel(f"TV média por token, texto web {lang}")
        a.set_ylabel(f"queda de acurácia {lang} (pp)")
        a.set_title(f"TV × perda em {lang}", fontsize=9, color=INK, loc="left")
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    return svg(fig)


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
@page :first { margin: 0; }
:root { --ink:#0b0b0b; --ink2:#52514e; --mut:#8a8984; --rule:#e4e3df; --bg:#fcfcfb; --acc:#2a78d6;
        --cov:#0f1720; --cov2:#1c2633; --pos:#008300; --neg:#c43c3b; }
* { box-sizing: border-box; }
body { font-family: "Inter", "DejaVu Sans", sans-serif; color: var(--ink); background: var(--bg);
       font-size: 9.6pt; line-height: 1.45; margin: 0; }
.cover { background: var(--cov); color: #e8ecf1; padding: 40mm 20mm 18mm; height: 297mm;
         page-break-after: always; margin: 0; overflow: hidden; }
.cover .tag { letter-spacing: .18em; font-size: 8pt; color: #8fa3b8; text-transform: uppercase; }
.cover h1 { font-size: 27pt; line-height: 1.12; margin: 10mm 0 4mm; font-weight: 700; }
.cover .sub { font-size: 11pt; color: #b9c6d4; max-width: 150mm; }
.cover .meta { margin-top: 12mm; font-size: 8.5pt; color: #8fa3b8; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm; margin: 14mm 0 0; }
.kpi { background: var(--cov2); border-radius: 3mm; padding: 4mm; }
.kpi .kv { font-size: 18pt; font-weight: 700; color: #fff; }
.kpi .kl { font-size: 8pt; color: #b9c6d4; margin-top: 1mm; }
.kpi .ks { font-size: 7pt; color: #8fa3b8; margin-top: 1.5mm; }
.body .kpis .kpi { background: #f1f0ec; } .body .kpi .kv { color: var(--ink); }
.body .kpi .kl { color: var(--ink2); } .body .kpi .ks { color: var(--mut); }
h2 { font-size: 15pt; margin: 9mm 0 2mm; border-bottom: 1.5px solid var(--ink); padding-bottom: 1.5mm;
     page-break-after: avoid; }
h2 .sec { color: var(--acc); margin-right: 2mm; }
h3 { font-size: 10.5pt; margin: 5mm 0 1.5mm; page-break-after: avoid; }
p { margin: 1.5mm 0 2.5mm; }
.lead { color: var(--ink2); }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 8.3pt;
        font-variant-numeric: tabular-nums; page-break-inside: avoid; }
th { text-align: left; font-weight: 600; color: var(--ink2); border-bottom: 1px solid var(--ink2);
     padding: 1.2mm 1.6mm; }
td { border-bottom: 1px solid var(--rule); padding: 1.1mm 1.6mm; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
table.left td, table.left th { text-align: left !important; }
.leitura { border-left: 3px solid var(--acc); background: #f1f5fb; padding: 2.5mm 4mm; margin: 3mm 0 4mm;
           page-break-inside: avoid; }
.leitura .lt { font-size: 7.5pt; letter-spacing: .15em; color: var(--acc); font-weight: 700; }
.leitura ul { margin: 1mm 0 0; padding-left: 4.5mm; } .leitura li { margin: .8mm 0; }
.fig { margin: 3mm 0; page-break-inside: avoid; } .fig svg { width: 100%; height: auto; }
.cap { font-size: 7.8pt; color: var(--mut); margin-top: -1mm; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; }
.verdict { display: inline-block; padding: .3mm 2mm; border-radius: 2mm; font-size: 7.5pt; font-weight: 700; }
.ok { background: #e3f3e3; color: var(--pos); } .no { background: #f8e4e4; color: var(--neg); }
.mix { background: #fbf1dc; color: #8a5a00; }
code { font-size: 8pt; background: #f1f0ec; padding: 0 1mm; border-radius: 1mm; }
.pb { page-break-before: always; }
.achado { display: grid; grid-template-columns: 9mm 1fr; gap: 3mm; border: 1px solid var(--rule); border-left: 4px solid var(--acc);
          border-radius: 2mm; padding: 3mm 4mm; margin: 3mm 0 4mm; page-break-inside: avoid; background: #fff; }
.achado .an { font-size: 20pt; font-weight: 700; color: var(--acc); line-height: 1; }
.achado h3 { margin: 0 0 1.5mm; }
.small { font-size: 8pt; color: var(--ink2); }
"""


def load_text():
    spec = importlib.util.spec_from_file_location("texto", OUT / "texto.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    e1, e2 = RES / "e1", RES / "e2"
    acc = pd.read_csv(e1 / "acc.csv", index_col=0)
    tests = pd.read_csv(e1 / "tests.csv")
    tv = pd.read_csv(e1 / "tv.csv", header=[0, 1], index_col=0) if (e1 / "tv.csv").exists() else pd.DataFrame()
    if len(tv):
        tv.columns = [f"{a}_{b}" for a, b in tv.columns]
    s = pd.read_csv(e2 / "serve.csv")
    sm = pd.read_csv(e2 / "serve_summary.csv")
    gg = pd.read_csv(e2 / "gguf.csv")
    qt = pd.read_csv(e2 / "quality_tests.csv")
    common = pd.read_csv(e2 / "acc_common_items.csv", index_col=0)
    T = load_text()
    figs = {"thr": fig_throughput(s), "energy": fig_energy(s)}
    figs["frontier"] = fig_frontier(T.frontier_points(common, gg))
    figs["drops"] = fig_drops(acc, T.SPECS, T.SPEC_NAMES)
    if len(tv):
        figs["tv"] = fig_tv(tv, acc, T.SPECS)
    ctx = dict(acc=acc, tests=tests, tv=tv, s=s, sm=sm, gg=gg, qt=qt, common=common, figs=figs,
               h4=json.loads((e1 / "summary.json").read_text()).get("h4", []),
               table=table, leitura=leitura, kpi=kpi, pct=pct, pp=pp, ci=ci, num=num)
    body = T.build(ctx)
    html = f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>" \
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>" \
           f"<title>Lumi-4B comprimido</title><style>{CSS}</style></head><body>{body}</body></html>"
    OUT.mkdir(exist_ok=True)
    (OUT / "lumi4b_compressao.html").write_text(html)
    from playwright.sync_api import sync_playwright
    chrome = next(Path.home().glob(".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/"
                                   "chrome-headless-shell"))
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(chrome))
        pg = b.new_page()
        pg.goto((OUT / "lumi4b_compressao.html").as_uri())
        pg.pdf(path=str(OUT / "lumi4b_compressao.pdf"), format="A4", print_background=True,
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:7pt;color:#8a8984;width:100%;text-align:center'>"
                               "Lumi-4B comprimido · E1 + E2 · <span class='pageNumber'></span>/"
                               "<span class='totalPages'></span></div>",
               prefer_css_page_size=True)
        b.close()
    print("[ok]", OUT / "lumi4b_compressao.pdf")


if __name__ == "__main__":
    main()
