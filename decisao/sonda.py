"""P: sonda linear nos estados internos como sinal de abstenção (protocolo em decisao/README.md).

  extrair (GPU): estados ocultos na posição da leitura -> /raid/$USER/lumi_sonda/{train,calib,dev}.npz + .jsonl
  analisar (CPU): sonda escolhida por CV no train, limiar no calib, avaliação no dev contra S1.
Roda só dentro do Slurm (decisao/sonda.sbatch).
"""
import argparse, json
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------------------------- extração
def cmd_extrair(a):
    import torch
    import lumi_extra as X
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    be, tpl = X.backend_native(), X.template()
    layers, norm = X.layers_and_norm(be)
    L = len(layers)
    keep = [int(x) - 1 for x in a.layers.split(",")]
    assert keep[-1] == L - 1, "a última camada precisa estar na lista (sanidade)"
    W = be._head_weight().float()
    caps = {}

    def hook(i):
        def f(m, inp, outp):
            caps[i] = outp[0] if isinstance(outp, tuple) else outp
        return f
    hs = [layers[i].register_forward_hook(hook(i)) for i in keep]
    sources = a.sources.split(",")
    meta = {"camadas": [k + 1 for k in keep], "contagens": {}}
    for split in ("train", "calib", "dev"):
        cs = []
        for s in sources:
            cs += X.cases_for(s, split, held_out="exclude", per_family=a.train_per_family) if split == "train" \
                else X.cases_for(s, split)
        H, rows = [], []
        for b in range(0, len(cs), a.batch):
            chunk = cs[b:b + a.batch]
            enc = [X.encode(be, tpl, c) for c in chunk]
            caps.clear()
            ids, mask, lens = X.pad_batch(be, [e[0] for e in enc])
            with torch.inference_mode():
                o = be._body()(input_ids=ids, attention_mask=mask)
            final = o[0] if isinstance(o, tuple) else o.last_hidden_state
            idx = torch.arange(len(chunk), device=be.device); last = lens - 1
            ref = torch.log_softmax(final[idx, last].float() @ W.T, -1)
            h = torch.stack([caps[k][idx, last] for k in keep], 1)          # (b, camadas, d), pré-norma
            chk = torch.log_softmax(norm(h[:, -1]).float() @ W.T, -1)
            for j, (c, (_, cands, labels)) in enumerate(zip(chunk, enc)):
                cc = torch.tensor(cands, device=be.device)
                diff = float((chk[j, cc] - ref[j, cc]).abs().max())
                if diff > a.tol:
                    raise SystemExit(f"SANIDADE FALHOU {c.id}: norma(h_L)·head ≠ leitura oficial (|Δ|={diff:.4f})")
                rows.append({"case_id": c.id, "group_id": c.group_id, "family": c.family, "source": c.source,
                             "held_out": c.held_out, "gold": c.gold, "labels": labels, "ref": ref[j, cc].tolist()})
            H.append(h.float().cpu().numpy().astype(np.float16))
        np.save(out / f"{split}.npy", np.concatenate(H))
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        meta["contagens"][split] = len(rows)
        print(split, len(rows), flush=True)
    for h in hs:
        h.remove()
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- análise
def auroc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oracle_min_risk(conf, err, min_accept=20):
    idx = np.argsort(-conf, kind="stable"); cum = np.cumsum(err[idx]) / np.arange(1, len(conf) + 1)
    return float(cum[min_accept - 1:].min()) if len(conf) >= min_accept else None


def cmd_analisar(a):
    import ordem_abstencao as d1
    from trajetoria import fit_lr, softmax_lp
    src_dir, out = Path(a.feats), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src_dir / "meta.json").read_text())

    def load(sp):
        rows = [json.loads(l) for l in (src_dir / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        H = np.load(src_dir / f"{sp}.npy").astype(np.float32)
        for r in rows:
            p = softmax_lp(r["ref"]); k = int(p.argmax())
            r["pred"], r["pmax"] = r["labels"][k], float(p[k])
        return rows, H
    (tr, Htr), (ca, Hca), (de, Hde) = load("train"), load("calib"), load("dev")
    assert not any(r["held_out"] for r in tr) and not any(r["held_out"] for r in ca)
    ytr = np.array([r["pred"] == r["gold"] for r in tr], float)
    lg = lambda rows: np.log(np.clip([r["pmax"] for r in rows], 1e-6, 1 - 1e-6)) - np.log1p(-np.clip([r["pmax"] for r in rows], 1e-6, 1 - 1e-6))

    groups = sorted({r["group_id"] for r in tr})
    fold_of = {g: k % 5 for k, g in enumerate(np.random.default_rng(0).permutation(groups))}
    fi = np.array([fold_of[r["group_id"]] for r in tr])

    def pca(l):
        X = Htr[:, l]; mu, sd = X.mean(0), X.std(0) + 1e-6
        _, _, Vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
        V = Vt[: a.pca].T
        return lambda H: ((H[:, l] - mu) / sd) @ V

    cv, proj = {}, {}
    for li, layer in enumerate(meta["camadas"]):
        proj[layer] = pca(li); Z = proj[layer](Htr)
        for lam in (0.1, 1.0, 10.0, 100.0):
            oof = np.zeros(len(tr))
            for k in range(5):
                f, _ = fit_lr(Z[fi != k], ytr[fi != k], l2=lam)
                oof[fi == k] = f(Z[fi == k])
            cv[(layer, lam)] = auroc(oof, ytr)
            print(f"CV train camada {layer} λ={lam}: AUROC {cv[(layer, lam)]:.3f}", flush=True)
    best_layer, best_lam = max(cv, key=cv.get)
    auroc_s1_train = auroc([r["pmax"] for r in tr], ytr)
    Ztr, Zca, Zde = proj[best_layer](Htr), proj[best_layer](Hca), proj[best_layer](Hde)
    fP, _ = fit_lr(Ztr, ytr, l2=best_lam)
    fPp, _ = fit_lr(np.c_[Ztr, lg(tr)], ytr, l2=best_lam)
    conf = {"S1": ([r["pmax"] for r in ca], [r["pmax"] for r in de]),
            "P": (fP(Zca), fP(Zde)),
            "P+": (fPp(np.c_[Zca, lg(ca)]), fPp(np.c_[Zde, lg(de)]))}

    res = {"protocolo": "decisao/README.md (P)", "cv_train": {f"{l}|{lam}": v for (l, lam), v in cv.items()},
           "escolha": {"camada": best_layer, "lambda": best_lam, "auroc_cv": cv[(best_layer, best_lam)],
                       "auroc_S1_train": auroc_s1_train}, "fontes": {}}
    lines = ["| fonte | grupo | sinal | AUROC méd. fam. | risco aceitos | cobertura | AURC | risco mín. oráculo | Δrisco vs S1 [IC95] | ΔAURC vs S1 [IC95] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    fm = lambda x: "—" if x is None else f"{100 * x:.1f}"
    for src in sorted({r["source"] for r in de}):
        ci_ = [i for i, r in enumerate(ca) if r["source"] == src]; di_ = [i for i, r in enumerate(de) if r["source"] == src]
        items = {s: ([{**ca[i], "conf": float(cc[i])} for i in ci_], [{**de[i], "conf": float(dc[i])} for i in di_])
                 for s, (cc, dc) in conf.items()}
        taus = {s: d1.threshold(np.array([i["conf"] for i in c]), np.array([i["pred"] != i["gold"] for i in c], float))
                for s, (c, _) in items.items()}
        sres = {"limiares": taus}
        for grp, flag in (("vistas", False), ("reservadas", True)):
            g = {}
            for s in conf:
                its = [i for i in items[s][1] if i["held_out"] == flag]
                if not its:
                    continue
                tab = d1.family_table(its, taus[s])
                fam_au, fam_or = {}, {}
                for f in tab:
                    fi_ = [i for i in its if i["family"] == f]
                    cf = np.array([i["conf"] for i in fi_]); er = np.array([i["pred"] != i["gold"] for i in fi_], float)
                    fam_au[f] = auroc(cf, er == 0); fam_or[f] = oracle_min_risk(cf, er)
                    tab[f]["auroc"], tab[f]["risco_min_oraculo"] = fam_au[f], fam_or[f]
                g[s] = {"auroc": d1.fam_mean(tab, "auroc"), "risco_aceitos": d1.fam_mean(tab, "risco_aceitos"),
                        "cobertura": d1.fam_mean(tab, "cobertura"), "aurc": d1.fam_mean(tab, "aurc"),
                        "risco_min_oraculo": d1.fam_mean(tab, "risco_min_oraculo"), "por_familia": tab}
            if "S1" in g:
                s1 = [i for i in items["S1"][1] if i["held_out"] == flag]
                for s in ("P", "P+"):
                    sb = [i for i in items[s][1] if i["held_out"] == flag]
                    g[s]["menos_S1"] = {k: {"delta": (g[s][k] - g["S1"][k]) if g[s][k] is not None and g["S1"][k] is not None else None,
                                            "ic95": d1.boot_diff(s1, sb, taus["S1"], taus[s], k)}
                                        for k in ("risco_aceitos", "cobertura", "aurc")}
            sres[grp] = g
            for s, v in g.items():
                ms = v.get("menos_S1", {})
                dd = lambda k: "" if k not in ms or ms[k]["delta"] is None else \
                    f"{100*ms[k]['delta']:+.1f} [{100*ms[k]['ic95'][0]:+.1f}; {100*ms[k]['ic95'][1]:+.1f}]"
                lines.append(f"| {src} | {grp} | {s} | {fm(v['auroc'])} | {fm(v['risco_aceitos'])} | {fm(v['cobertura'])} | "
                             f"{fm(v['aurc'])} | {fm(v['risco_min_oraculo'])} | {dd('risco_aceitos')} | {dd('aurc')} |")
        res["fontes"][src] = sres
    (out / "sonda.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "sonda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(res["escolha"], indent=1)); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extrair"); e.add_argument("--out", required=True); e.add_argument("--sources", default="felhen,telepatia")
    e.add_argument("--layers", default="12,20,28,36,42"); e.add_argument("--batch", type=int, default=8)
    e.add_argument("--train-per-family", type=int, default=600); e.add_argument("--tol", type=float, default=0.02)
    n = sub.add_parser("analisar"); n.add_argument("--feats", required=True); n.add_argument("--out", required=True)
    n.add_argument("--pca", type=int, default=256)
    a = ap.parse_args()
    {"extrair": cmd_extrair, "analisar": cmd_analisar}[a.cmd](a)


if __name__ == "__main__":
    main()
