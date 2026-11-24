"""D1: desacordo entre ordens como sinal de abstenção (protocolo em decisao/README.md).

Lê as execuções do E1 do lumi-decision (predicoes.jsonl, uma linha por caso x ordem) e compara S1..S4.
Usa as funções do próprio grupo (lumi_decision.metrics) para temperatura e limiar de S1, e a mesma regra
de corte para os outros sinais. Roda só dentro do Slurm (slurm/d1_analise.sbatch).

  python decisao/ordem_abstencao.py --runs /raid/$USER/lumi-decision/runs --prefix e1_lumi_27b30f4_nativa \
      --out decisao/resultados/d1
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from lumi_decision import metrics  # código do grupo (src/ no PYTHONPATH do venv deles)

TARGET = 0.10
SIGNALS = ("S1", "S2", "S3", "S4")


def load(run: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run / "predicoes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r.setdefault("pred", max(r["probs"], key=r["probs"].get))
    return rows


def pair_cases(rows):
    """case_id -> {"canonical": row, "reversed": row}; só casos com as duas ordens."""
    by = defaultdict(dict)
    for r in rows:
        by[r["case_id"]][r["order"]] = r
    return {c: o for c, o in by.items() if {"canonical", "reversed"} <= set(o)}


def case_signals(o: dict) -> dict:
    """Decisão e confiança de S2..S4 a partir das duas ordens (probs já com temperatura)."""
    pc, pr = o["canonical"]["probs"], o["reversed"]["probs"]
    assert set(pc) == set(pr), "opções diferentes entre ordens"
    pbar = {k: (pc[k] + pr[k]) / 2 for k in pc}
    dec = max(pbar, key=pbar.get)
    agree = max(pc, key=pc.get) == max(pr, key=pr.get)
    base = o["canonical"]
    return {"case_id": base["case_id"], "group_id": base["group_id"], "family": base["family"],
            "source": base["source"], "held_out": base["held_out"], "gold": base["gold"], "pred": dec,
            "agree": agree,
            "conf": {"S2": pbar[dec], "S3": pbar[dec] if agree else 0.0, "S4": min(pc[dec], pr[dec])}}


def s1_items(rows):
    """S1: cada (caso, ordem) é um item; confiança = max p_T da ordem."""
    return [{"case_id": r["case_id"], "group_id": r["group_id"], "family": r["family"], "source": r["source"],
             "held_out": r["held_out"], "gold": r["gold"], "pred": max(r["probs"], key=r["probs"].get),
             "conf": max(r["probs"].values()), "order": r["order"]} for r in rows]


def threshold(conf, err, target=TARGET, min_accept=20):
    """Mesma regra de metrics.threshold_for_risk, sobre arrays."""
    idx = np.argsort(-conf, kind="stable")
    cum = np.cumsum(err[idx]) / np.arange(1, len(conf) + 1)
    sc = conf[idx]
    cuts = [i for i in range(min_accept - 1, len(conf)) if i == len(conf) - 1 or sc[i] > sc[i + 1]]
    ok = [i for i in cuts if cum[i] <= target]
    return float(sc[max(ok)]) if ok else None


def aurc(conf, err):
    idx = np.argsort(-conf, kind="stable")
    return float(np.mean(np.cumsum(err[idx]) / np.arange(1, len(conf) + 1)))


def risk_at_cov(conf, err, cov=0.5):
    idx = np.argsort(-conf, kind="stable")
    k = max(1, int(round(cov * len(conf))))
    return float(err[idx[:k]].mean())


def family_table(items, tau):
    """Por família: n, cobertura e risco entre aceitos com limiar fixo; AURC; risco@50%."""
    fam = defaultdict(list)
    for it in items:
        fam[it["family"]].append(it)
    out = {}
    for f, its in sorted(fam.items()):
        conf = np.array([i["conf"] for i in its]); err = np.array([i["pred"] != i["gold"] for i in its], float)
        acc = conf >= tau if tau is not None else np.zeros(len(its), bool)
        out[f] = {"n": len(its), "cobertura": float(acc.mean()),
                  "risco_aceitos": float(err[acc].mean()) if acc.any() else None,
                  "erro_sem_abster": float(err.mean()), "aurc": aurc(conf, err), "risco_cob50": risk_at_cov(conf, err)}
    return out


def fam_mean(tab, key):
    vals = [v[key] for v in tab.values() if v[key] is not None]
    return float(np.mean(vals)) if vals else None


def boot_diff(items_a, items_b, tau_a, tau_b, key, n_boot=2000, seed=0):
    """IC da diferença B − A da média por família de `key`, reamostrando grupos dentro de cada família
    (os dois sinais usam os MESMOS grupos sorteados). key: risco_aceitos | aurc | cobertura."""
    def by_fam_group(items):
        d = defaultdict(lambda: defaultdict(list))
        for i in items:
            d[i["family"]][i["group_id"]].append(i)
        return d
    A, B = by_fam_group(items_a), by_fam_group(items_b)
    fams = sorted(set(A) & set(B))
    rng = np.random.default_rng(seed)

    def stat(its, tau):
        conf = np.array([i["conf"] for i in its]); err = np.array([i["pred"] != i["gold"] for i in its], float)
        if key == "aurc":
            return aurc(conf, err)
        acc = conf >= tau if tau is not None else np.zeros(len(its), bool)
        if key == "cobertura":
            return float(acc.mean())
        return float(err[acc].mean()) if acc.any() else np.nan

    boots = []
    for _ in range(n_boot):
        da, db = [], []
        for f in fams:
            groups = sorted(set(A[f]) & set(B[f]))
            pick = [groups[j] for j in rng.integers(0, len(groups), len(groups))]
            da.append(stat([i for g in pick for i in A[f][g]], tau_a))
            db.append(stat([i for g in pick for i in B[f][g]], tau_b))
        boots.append(np.nanmean(db) - np.nanmean(da))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prefix", required=True, help="ex.: e1_lumi_27b30f4_nativa")
    ap.add_argument("--scope", default="completo")
    ap.add_argument("--sources", default="felhen,telepatia,traces")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"protocolo": "decisao/README.md", "risco_alvo": TARGET, "fontes": {}}

    for src in a.sources.split(","):
        calib = load(Path(a.runs) / f"{a.prefix}_{src}_calib_{a.scope}")
        dev = load(Path(a.runs) / f"{a.prefix}_{src}_dev_{a.scope}")
        calib_seen = [r for r in calib if not r["held_out"]]
        T = metrics.fit_temperature(calib_seen)                       # igual ao lumi-eval calibrar
        cT, dT = metrics.with_temperature(calib_seen, T), metrics.with_temperature(dev, T)
        tau_group = metrics.threshold_for_risk(cT, TARGET)            # limiar do grupo, para conferência

        items = {"S1": {"calib": s1_items(cT), "dev": s1_items(dT)}}
        cs_c = [case_signals(o) for o in pair_cases(cT).values()]
        cs_d = [case_signals(o) for o in pair_cases(dT).values()]
        for s in ("S2", "S3", "S4"):
            items[s] = {sp: [{**c, "conf": c["conf"][s]} for c in cs] for sp, cs in (("calib", cs_c), ("dev", cs_d))}

        taus = {}
        for s in SIGNALS:
            c = items[s]["calib"]
            taus[s] = threshold(np.array([i["conf"] for i in c]), np.array([i["pred"] != i["gold"] for i in c], float))
        assert taus["S1"] == tau_group, f"S1 diverge do limiar do grupo: {taus['S1']} != {tau_group}"

        sres = {"temperatura": T, "limiares": taus, "n_casos_dev": len(cs_d)}
        for grp, flag in (("vistas", False), ("reservadas", True)):
            g = {}
            for s in SIGNALS:
                its = [i for i in items[s]["dev"] if i["held_out"] == flag]
                if not its:
                    continue
                tab = family_table(its, taus[s])
                g[s] = {"risco_aceitos": fam_mean(tab, "risco_aceitos"), "cobertura": fam_mean(tab, "cobertura"),
                        "aurc": fam_mean(tab, "aurc"), "risco_cob50": fam_mean(tab, "risco_cob50"),
                        "erro_sem_abster": fam_mean(tab, "erro_sem_abster"), "por_familia": tab}
            if "S1" in g:
                s1 = [i for i in items["S1"]["dev"] if i["held_out"] == flag]
                # fração das decisões aceitas por S1 que mudam ao inverter as opções
                pred = {(i["case_id"], i["order"]): i["pred"] for i in s1}
                acc_s1 = [i for i in s1 if taus["S1"] is not None and i["conf"] >= taus["S1"]]
                other = {"canonical": "reversed", "reversed": "canonical"}
                fl = [pred[(i["case_id"], i["order"])] != pred.get((i["case_id"], other[i["order"]])) for i in acc_s1
                      if (i["case_id"], other[i["order"]]) in pred]
                g["S1"]["aceitas_que_mudam_com_ordem"] = float(np.mean(fl)) if fl else None
                for s in ("S2", "S3", "S4"):
                    if s in g:
                        sb = [i for i in items[s]["dev"] if i["held_out"] == flag]
                        g[s]["menos_S1"] = {k: {"delta": (g[s][k] - g["S1"][k]) if g[s][k] is not None and g["S1"][k] is not None else None,
                                                "ic95": boot_diff(s1, sb, taus["S1"], taus[s], k)}
                                            for k in ("risco_aceitos", "cobertura", "aurc")}
                if "S3" in g and "S2" in g:
                    s2 = [i for i in items["S2"]["dev"] if i["held_out"] == flag]
                    s3 = [i for i in items["S3"]["dev"] if i["held_out"] == flag]
                    g["S3"]["menos_S2"] = {k: boot_diff(s2, s3, taus["S2"], taus["S3"], k) for k in ("risco_aceitos", "cobertura")}
            sres[grp] = g
        res["fontes"][src] = sres

    (out / "d1.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["| fonte | grupo | sinal | risco aceitos (alvo 10%) | cobertura | AURC | risco@50% | Δrisco vs S1 [IC95] |", "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "—" if x is None else f"{100 * x:.1f}"
    for src, sres in res["fontes"].items():
        for grp in ("vistas", "reservadas"):
            for s, v in sres.get(grp, {}).items():
                d = v.get("menos_S1", {}).get("risco_aceitos")
                dd = "" if not d or d["delta"] is None else f"{100*d['delta']:+.1f} [{100*d['ic95'][0]:+.1f}; {100*d['ic95'][1]:+.1f}]"
                lines.append(f"| {src} | {grp} | {s} | {f(v['risco_aceitos'])} | {f(v['cobertura'])} | {f(v['aurc'])} | {f(v['risco_cob50'])} | {dd} |")
    (out / "d1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
