"""T (trajetória da decisão pelas camadas como sinal de abstenção) e R (rótulos para uma família nova).
Protocolos em decisao/README.md. Só CPU; roda dentro do Slurm (decisao/traj_rotulos.sbatch).

  python decisao/trajetoria.py traj --dump decisao/resultados/d2 --out decisao/resultados/traj
  python decisao/trajetoria.py rotulos --runs /raid/$USER/lumi-decision/runs --prefix e1_lumi_27b30f4_nativa \
      --out decisao/resultados/rotulos
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

import ordem_abstencao as d1

TARGET = 0.10


def softmax_lp(lp):
    a = np.asarray(lp, float)
    a = np.exp(a - a.max())
    return a / a.sum()


# ----------------------------------------------------------------------------------------------- T
def traj_item(r, lo=30):
    L = len(r["logit"])
    p_ref = softmax_lp(r["ref"])
    fin = int(p_ref.argmax())
    decs = [int(np.argmax(x)) for x in r["logit"]]
    l_star = L  # primeira camada (1..42) a partir da qual o argmax não muda mais e é a decisão final
    for l in range(L - 1, -1, -1):
        if decs[l] != fin:
            break
        l_star = l + 1
    trocas = sum(decs[l] != decs[l - 1] for l in range(lo, L))
    pfin = float(np.mean([softmax_lp(r["logit"][l])[fin] for l in range(lo - 1, L)]))
    pmax = float(p_ref[fin])
    return {"case_id": r["case_id"], "group_id": r["group_id"], "family": r["family"], "source": r["source"],
            "held_out": r["held_out"], "gold": r["gold"], "pred": r["labels"][fin],
            "pmax": pmax, "l_star": l_star, "trocas": trocas, "pfin": pfin, "L": L}


def feats(its):
    return np.array([[i["pmax"], i["l_star"] / i["L"], i["trocas"], i["pfin"]] for i in its], float)


def fit_lr(X, y, l2=1e-2, iters=100):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, Z.T @ (p - y) + l2 * w)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Xn: 1 / (1 + np.exp(-np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w)), w


def cmd_traj(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    load = lambda sp: [traj_item(json.loads(l)) for l in (Path(a.dump) / f"camadas_{sp}.jsonl").read_text().splitlines() if l.strip()]
    calib_all, dev_all = load("calib"), load("dev")
    assert not any(i["held_out"] for i in calib_all), "calib do dump deveria ter só vistas"
    res = {"protocolo": "decisao/README.md (T)", "risco_alvo": TARGET, "fontes": {}}
    lines = ["| fonte | grupo | sinal | risco aceitos | cobertura | AURC | Δrisco vs S1 [IC95] | ΔAURC vs S1 [IC95] |",
             "|---|---|---|---|---|---|---|---|"]
    for src in sorted({i["source"] for i in dev_all}):
        cal = [i for i in calib_all if i["source"] == src]
        dev = [i for i in dev_all if i["source"] == src]
        ycal = np.array([i["pred"] == i["gold"] for i in cal], float)
        # T2: predições fora da dobra no calib (5 dobras por grupo); modelo final em todo o calib para o dev
        groups = sorted({i["group_id"] for i in cal})
        fold = {g: k % 5 for k, g in enumerate(np.random.default_rng(0).permutation(groups))}
        fi = np.array([fold[i["group_id"]] for i in cal])
        Xc, Xd = feats(cal), feats(dev)
        oof = np.zeros(len(cal))
        for k in range(5):
            f, _ = fit_lr(Xc[fi != k], ycal[fi != k])
            oof[fi == k] = f(Xc[fi == k])
        f_all, w = fit_lr(Xc, ycal)
        pdev = f_all(Xd)
        conf = {"S1": ([i["pmax"] for i in cal], [i["pmax"] for i in dev]),
                "T1": ([(i["L"] - i["l_star"]) + i["pmax"] for i in cal], [(i["L"] - i["l_star"]) + i["pmax"] for i in dev]),
                "T2": (list(oof), list(pdev))}
        items = {s: ([{**i, "conf": float(c)} for i, c in zip(cal, cc)], [{**i, "conf": float(c)} for i, c in zip(dev, dc)])
                 for s, (cc, dc) in conf.items()}
        taus = {s: d1.threshold(np.array([i["conf"] for i in ci]), np.array([i["pred"] != i["gold"] for i in ci], float))
                for s, (ci, _) in items.items()}
        sres = {"limiares": taus, "pesos_T2": [float(x) for x in w], "n_calib": len(cal), "n_dev": len(dev),
                "acerto_dev": {g: float(np.mean([i["pred"] == i["gold"] for i in dev if i["held_out"] == fl]))
                               for g, fl in (("vistas", False), ("reservadas", True)) if any(i["held_out"] == fl for i in dev)},
                # diagnóstico: l* médio para acertos e erros
                "l_star_medio": {g: {"acerto": float(np.mean([i["l_star"] for i in dev if i["held_out"] == fl and i["pred"] == i["gold"]] or [np.nan])),
                                     "erro": float(np.mean([i["l_star"] for i in dev if i["held_out"] == fl and i["pred"] != i["gold"]] or [np.nan]))}
                                 for g, fl in (("vistas", False), ("reservadas", True))}}
        for grp, flag in (("vistas", False), ("reservadas", True)):
            g = {}
            for s in ("S1", "T1", "T2"):
                its = [i for i in items[s][1] if i["held_out"] == flag]
                if not its:
                    continue
                tab = d1.family_table(its, taus[s])
                g[s] = {"risco_aceitos": d1.fam_mean(tab, "risco_aceitos"), "cobertura": d1.fam_mean(tab, "cobertura"),
                        "aurc": d1.fam_mean(tab, "aurc"), "risco_cob50": d1.fam_mean(tab, "risco_cob50"), "por_familia": tab}
            if "S1" in g:
                s1 = [i for i in items["S1"][1] if i["held_out"] == flag]
                for s in ("T1", "T2"):
                    sb = [i for i in items[s][1] if i["held_out"] == flag]
                    g[s]["menos_S1"] = {k: {"delta": (g[s][k] - g["S1"][k]) if g[s][k] is not None and g["S1"][k] is not None else None,
                                            "ic95": d1.boot_diff(s1, sb, taus["S1"], taus[s], k)}
                                        for k in ("risco_aceitos", "cobertura", "aurc")}
            sres[grp] = g
            fm = lambda x: "—" if x is None else f"{100 * x:.1f}"
            for s, v in g.items():
                ms = v.get("menos_S1", {})
                dd = lambda k: "" if k not in ms or ms[k]["delta"] is None else \
                    f"{100*ms[k]['delta']:+.1f} [{100*ms[k]['ic95'][0]:+.1f}; {100*ms[k]['ic95'][1]:+.1f}]"
                lines.append(f"| {src} | {grp} | {s} | {fm(v['risco_aceitos'])} | {fm(v['cobertura'])} | {fm(v['aurc'])} | {dd('risco_aceitos')} | {dd('aurc')} |")
        res["fontes"][src] = sres
    (out / "traj.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "traj.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({s: {k: v[k] for k in ("acerto_dev", "l_star_medio", "limiares")} for s, v in res["fontes"].items()}, indent=1))


# ----------------------------------------------------------------------------------------------- R
def ltt(conf, err, alpha, delta):
    """Learn-then-Test, sequência fixa do limiar mais alto ao mais baixo (igual ao D4 corrigido)."""
    from scipy.stats import binom
    n_min = int(np.ceil(np.log(delta) / np.log(1 - alpha)))
    idx = np.argsort(-conf, kind="stable"); sc = conf[idx]
    ce = np.cumsum(err[idx]); n = np.arange(1, len(conf) + 1)
    cuts = np.array([i for i in range(len(conf)) if i == len(conf) - 1 or sc[i] > sc[i + 1]], int)
    cv = cuts[n[cuts] >= n_min]
    if len(cv) == 0:
        return None
    p = binom.cdf(ce[cv], n[cv], alpha)
    fail = np.nonzero(p > delta)[0]
    last = (fail[0] - 1) if len(fail) else len(cv) - 1
    return float(sc[cv[last]]) if last >= 0 else None


def cmd_rotulos(a):
    from lumi_decision import metrics
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    budgets = [int(x) for x in a.budgets.split(",")]
    rng = np.random.default_rng(0)
    res = {"protocolo": "decisao/README.md (R)", "alfa": a.alpha, "delta": a.delta, "sorteios": a.reps, "fontes": {}}
    lines = ["| fonte | família | sinal | m | estratégia | risco aceitos | cobertura | P(risco>α) | sem aceitos |",
             "|---|---|---|---|---|---|---|---|---|"]
    for src in a.sources.split(","):
        calib = [r for r in d1.load(Path(a.runs) / f"{a.prefix}_{src}_calib_completo") if not r["held_out"]]
        dev = d1.load(Path(a.runs) / f"{a.prefix}_{src}_dev_completo")
        T = metrics.fit_temperature(calib)
        cT, dT = metrics.with_temperature(calib, T), metrics.with_temperature(dev, T)
        sig = {"S1": ([i for i in d1.s1_items(cT) if i["order"] == "canonical"],
                      [i for i in d1.s1_items(dT) if i["order"] == "canonical"])}
        cc = [d1.case_signals(o) for o in d1.pair_cases(cT).values()]
        dd = [d1.case_signals(o) for o in d1.pair_cases(dT).values()]
        sig["S3"] = ([{**c, "conf": c["conf"]["S3"]} for c in cc], [{**c, "conf": c["conf"]["S3"]} for c in dd])
        sres = {"temperatura": T}
        for s, (ci, di) in sig.items():
            arr = lambda its: (np.array([i["conf"] for i in its]), np.array([i["pred"] != i["gold"] for i in its], float))
            c_cal, e_cal = arr(ci)
            tau_v = {"vistas_emp": d1.threshold(c_cal, e_cal), "vistas_ltt": ltt(c_cal, e_cal, a.alpha, a.delta)}
            fams = sorted({i["family"] for i in di if i["held_out"]})
            sres[s] = {"limiares_vistas": tau_v, "familias": {}}
            for fam in fams:
                its = [i for i in di if i["family"] == fam]
                byg = defaultdict(list)
                for i in its:
                    byg[i["group_id"]].append(i)
                gids = sorted(byg)
                fres = {"n": len(its)}
                for m in budgets:
                    if len(its) - m < a.min_test:
                        continue
                    acc_stats = defaultdict(lambda: {"risco": [], "cob": [], "viola": [], "vazio": []})
                    for _ in range(a.reps):
                        perm = rng.permutation(len(gids)); new, test, cnt = [], [], 0
                        for j in perm:
                            (new if cnt < m else test).extend(byg[gids[j]])
                            if cnt < m:
                                cnt += len(byg[gids[j]])
                        cn, en = arr(new); ct, et = arr(test)
                        taus = {**tau_v,
                                "novos_emp": d1.threshold(cn, en, min_accept=min(20, len(cn) // 2)),
                                "novos_ltt": ltt(cn, en, a.alpha, a.delta)}
                        for st, tau in taus.items():
                            acc = ct >= tau if tau is not None else np.zeros(len(ct), bool)
                            r = float(et[acc].mean()) if acc.any() else np.nan
                            S = acc_stats[st]
                            S["risco"].append(r); S["cob"].append(float(acc.mean()))
                            S["viola"].append(bool(acc.any() and r > a.alpha)); S["vazio"].append(not acc.any())
                    fres[str(m)] = {}
                    for st, S in acc_stats.items():
                        rr = np.array(S["risco"])
                        v = {"risco_aceitos": float(np.nanmean(rr)) if np.isfinite(rr).any() else None,
                             "cobertura": float(np.mean(S["cob"])), "p_viola": float(np.mean(S["viola"])),
                             "sem_aceitos": float(np.mean(S["vazio"]))}
                        fres[str(m)][st] = v
                        fm = lambda x: "—" if x is None else f"{100 * x:.1f}"
                        lines.append(f"| {src} | {fam} | {s} | {m} | {st} | {fm(v['risco_aceitos'])} | {fm(v['cobertura'])} | "
                                     f"{fm(v['p_viola'])} | {fm(v['sem_aceitos'])} |")
                sres[s]["familias"][fam] = fres
        res["fontes"][src] = sres
    (out / "rotulos.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "rotulos.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("traj"); t.add_argument("--dump", required=True); t.add_argument("--out", required=True)
    r = sub.add_parser("rotulos"); r.add_argument("--runs", required=True); r.add_argument("--prefix", required=True)
    r.add_argument("--out", required=True); r.add_argument("--sources", default="felhen,telepatia")
    r.add_argument("--budgets", default="25,50,100,200,400"); r.add_argument("--reps", type=int, default=200)
    r.add_argument("--min-test", type=int, default=50)
    r.add_argument("--alpha", type=float, default=0.10); r.add_argument("--delta", type=float, default=0.10)
    a = ap.parse_args()
    {"traj": cmd_traj, "rotulos": cmd_rotulos}[a.cmd](a)


if __name__ == "__main__":
    main()
