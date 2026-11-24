"""D2 (saída antecipada), D3 (reaproveitar o estado do contexto) e D4 (abstenção com garantia).

Protocolos em decisao/README.md. Usa os casos, prompts, template e backend do lumi-decision (implementação
nativa do transformers). Roda só dentro do Slurm (decisao/d2_saida.sbatch, d3_estado.sbatch, d4_conformal.sbatch).

  python decisao/lumi_extra.py saida    --out DIR [--sources felhen,telepatia]
  python decisao/lumi_extra.py estado   --out DIR
  python decisao/lumi_extra.py conformal --runs RUNS --prefix PREFIX --out DIR
"""
import argparse, copy, json, math, time
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL, REV = "ceia-br/lumi-4b", "27b30f40b930fd05cee5e1c1f8790f70fa5b816d"


# ----------------------------------------------------------------------------------------------- comum
def load_group():
    from lumi_decision import evaluate, prompts
    from lumi_decision.backends import make_backend
    return evaluate, prompts, make_backend


def cases_for(source, split, held_out="include", per_family=0, seed=0):
    evaluate, _, _ = load_group()
    ns = argparse.Namespace(source=source, split=split, families=None, held_out=held_out,
                            per_family=per_family, limit=0, seed=seed)
    return evaluate.select_cases(ns)


def backend_native():
    _, _, make_backend = load_group()
    return make_backend("hf", model=MODEL, revision=REV, dtype="bfloat16", trust_remote_code=False)


def template():
    evaluate, prompts, _ = load_group()
    p = Path(evaluate.ARTIFACTS_DIR) / "lumi" / "chat_template.jinja"
    return prompts.load_template(p)


def encode(be, tpl, case, order="canonical"):
    _, prompts, _ = load_group()
    opts = prompts.order_options(case, order)
    text = prompts.render(tpl, case, opts)
    ids, cands = prompts.candidate_tokens(be.tokenize, text, len(opts))
    return ids, cands, [o[0] for o in opts]


def layers_and_norm(be):
    import torch.nn as nn
    body = be._body()
    n = be.model.config.num_hidden_layers
    layers = next(m for m in body.children() if isinstance(m, nn.ModuleList) and len(m) == n)
    norm = next(m for name, m in body.named_children() if "norm" in name.lower())
    return layers, norm


def pad_batch(be, batch_ids):
    import torch
    lens = torch.tensor([len(x) for x in batch_ids], device=be.device)
    ids = torch.full((len(batch_ids), int(lens.max())), be.pad_id, dtype=torch.long, device=be.device)
    mask = torch.zeros_like(ids)
    for i, x in enumerate(batch_ids):
        ids[i, :len(x)] = torch.tensor(x, device=be.device)
        mask[i, :len(x)] = 1
    return ids, mask, lens


# ----------------------------------------------------------------------------------------------- D2
def cmd_saida(a):
    import torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    be, tpl = backend_native(), template()
    layers, norm = layers_and_norm(be)
    L = len(layers)
    W_head = be._head_weight().float()
    d = W_head.shape[1]
    caps = {}

    def hook(i):
        def f(m, inp, outp):
            caps[i] = outp[0] if isinstance(outp, tuple) else outp
        return f
    hs = [layers[i].register_forward_hook(hook(i)) for i in range(L)]

    def forward(batch_ids):
        caps.clear()
        ids, mask, lens = pad_batch(be, batch_ids)
        with torch.inference_mode():
            outp = be._body()(input_ids=ids, attention_mask=mask)
        final = outp[0] if isinstance(outp, tuple) else outp.last_hidden_state
        return lens, final

    sources = a.sources.split(",")
    # 1) tuned lens: ridge afim h_l -> h_L (pré-norma) nas últimas K posições, casos de TREINO (nunca vistos
    #    pelo modelo congelado e fora da avaliação). Acumula X^T X e X^T Y por camada na GPU.
    K = a.positions
    XtX = [torch.zeros(d + 1, d + 1, device=be.device) for _ in range(L)]
    XtY = [torch.zeros(d + 1, d, device=be.device) for _ in range(L)]
    train = []
    for s in sources:
        cs = cases_for(s, "train", held_out="exclude", per_family=a.train_per_family)
        train += cs
    rng = np.random.default_rng(0)
    rng.shuffle(train)
    train = train[: a.train_max]
    n_tok = 0
    for b in range(0, len(train), a.batch):
        batch = [encode(be, tpl, c)[0] for c in train[b:b + a.batch]]
        lens, _ = forward(batch)
        rows = [(i, p) for i, ln in enumerate(lens.tolist()) for p in range(max(0, ln - K), ln)]
        bi = torch.tensor([r[0] for r in rows], device=be.device); pi = torch.tensor([r[1] for r in rows], device=be.device)
        Y = caps[L - 1][bi, pi].float()
        for l in range(L):
            X = torch.cat([caps[l][bi, pi].float(), torch.ones(len(rows), 1, device=be.device)], 1)
            XtX[l] += X.T @ X; XtY[l] += X.T @ Y
        n_tok += len(rows)
    lens_W = []
    for l in range(L):
        lam = a.ridge * torch.diagonal(XtX[l]).mean()
        lens_W.append(torch.linalg.solve(XtX[l] + lam * torch.eye(d + 1, device=be.device), XtY[l]))
    del XtX, XtY
    torch.cuda.empty_cache()

    # 2) por camada: log-probs das letras pelo logit lens e pelo tuned lens, em calib e dev (ordem canônica)
    def run_split(split, fh):
        n = 0
        for s in sources:
            cs = cases_for(s, split)
            for b in range(0, len(cs), a.batch):
                chunk = cs[b:b + a.batch]
                enc = [encode(be, tpl, c) for c in chunk]
                lens, final = forward([e[0] for e in enc])
                idx = torch.arange(len(chunk), device=be.device); last = lens - 1
                ref = torch.log_softmax(final[idx, last].float() @ W_head.T, -1)   # leitura oficial (backbone c/ norma)
                for j, (c, (ids, cands, labels)) in enumerate(zip(chunk, enc)):
                    cc = torch.tensor(cands, device=be.device)
                    row = {"case_id": c.id, "group_id": c.group_id, "family": c.family, "source": c.source,
                           "held_out": c.held_out, "gold": c.gold, "labels": labels,
                           "ref": ref[j, cc].tolist(), "logit": [], "tuned": []}
                    for l in range(L):
                        h = caps[l][j, last[j]].float()
                        hl = norm(h.to(final.dtype)).float()
                        ht = norm((torch.cat([h, h.new_ones(1)]) @ lens_W[l]).to(final.dtype)).float()
                        row["logit"].append(torch.log_softmax(hl @ W_head.T, -1)[cc].tolist())
                        row["tuned"].append(torch.log_softmax(ht @ W_head.T, -1)[cc].tolist())
                    # sanidade: a última camada pelo logit lens tem de reproduzir a leitura oficial
                    diff = max(abs(x - y) for x, y in zip(row["logit"][-1], row["ref"]))
                    if diff > a.tol:
                        raise SystemExit(f"SANIDADE FALHOU {c.id}: camada final ≠ leitura oficial (|Δ|={diff:.4f})")
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
        return n

    counts = {}
    for split in ("calib", "dev"):
        with open(out / f"camadas_{split}.jsonl", "w", encoding="utf-8") as fh:
            counts[split] = run_split(split, fh)
    for h in hs:
        h.remove()

    # 3) custo por camada: tempo de GPU de cada bloco (eventos CUDA) num lote típico do dev
    t_layer = measure_layer_time(be, layers, [encode(be, tpl, c)[0] for c in cases_for(sources[0], "dev")[:a.batch]])
    meta = {"modelo": MODEL, "revisao": REV, "n_camadas": L, "tokens_tuned_lens": n_tok, "casos_treino_lens": len(train),
            "ridge": a.ridge, "posicoes": K, "contagens": counts, "tempo_camada_ms": t_layer,
            "tipos_camada": getattr(be.model.config, "hybrid_override_pattern", None)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    analyze_saida(out, meta)


def measure_layer_time(be, layers, batch_ids, reps=5):
    import torch
    ev = {}

    def pre(i):
        def f(m, inp):
            e = torch.cuda.Event(enable_timing=True); e.record(); ev.setdefault(i, []).append([e, None])
        return f

    def post(i):
        def f(m, inp, outp):
            e = torch.cuda.Event(enable_timing=True); e.record(); ev[i][-1][1] = e
        return f
    hs = [layers[i].register_forward_pre_hook(pre(i)) for i in range(len(layers))]
    hs += [layers[i].register_forward_hook(post(i)) for i in range(len(layers))]
    ids, mask, _ = pad_batch(be, batch_ids)
    with torch.inference_mode():
        for _ in range(reps + 1):
            be._body()(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize()
    for h in hs:
        h.remove()
    return [float(np.mean([s.elapsed_time(e) for s, e in ev[i][1:]])) for i in range(len(layers))]


def analyze_saida(out, meta, eps=(0.01, 0.02)):
    """Por camada: acerto e concordância com a decisão final. Política de saída: sai na primeira camada l (>= 8)
    cuja prob. máxima (tuned lens) passa o limiar τ_l escolhido no calib para que, entre os que saem em l, no
    máximo ε mudem de decisão em relação ao modelo completo. Avaliada no dev, por família (vistas/reservadas)."""
    L = meta["n_camadas"]
    cost = np.cumsum(meta["tempo_camada_ms"]); cost = cost / cost[-1]
    rows = {sp: [json.loads(x) for x in open(out / f"camadas_{sp}.jsonl", encoding="utf-8")] for sp in ("calib", "dev")}

    def dec(lp):
        return int(np.argmax(lp))

    res = {"por_camada": {}, "politica": {}}
    for lens in ("logit", "tuned"):
        per = []
        for l in range(L):
            for sp in ("dev",):
                for grp, flag in (("vistas", False), ("reservadas", True)):
                    rs = [r for r in rows[sp] if r["held_out"] == flag]
                    if not rs:
                        continue
                    agree = np.mean([dec(r[lens][l]) == dec(r["ref"]) for r in rs])
                    acc = np.mean([r["labels"][dec(r[lens][l])] == r["gold"] for r in rs])
                    per.append({"camada": l + 1, "grupo": grp, "concorda_final": float(agree), "acerto": float(acc)})
        res["por_camada"][lens] = per
    for e in eps:
        taus = {}
        cal = rows["calib"]
        for l in range(7, L - 1):
            conf = np.array([max(np.exp(r["tuned"][l]) / np.exp(r["tuned"][l]).sum()) for r in cal])
            flip = np.array([dec(r["tuned"][l]) != dec(r["ref"]) for r in cal], float)
            o = np.argsort(-conf); cum = np.cumsum(flip[o]) / np.arange(1, len(o) + 1)
            ok = [i for i in range(19, len(o)) if cum[i] <= e and (i == len(o) - 1 or conf[o][i] > conf[o][i + 1])]
            taus[l] = float(conf[o][max(ok)]) if ok else None
        pol = {}
        for grp, flag in (("vistas", False), ("reservadas", True)):
            rs = [r for r in rows["dev"] if r["held_out"] == flag]
            if not rs:
                continue
            exit_l, flips, accs = [], [], []
            for r in rs:
                l_exit = L - 1
                for l in range(7, L - 1):
                    t = taus.get(l)
                    p = np.exp(r["tuned"][l]); p = p / p.sum()
                    if t is not None and p.max() >= t:
                        l_exit = l; break
                lp = r["tuned"][l_exit] if l_exit < L - 1 else r["ref"]
                exit_l.append(l_exit); flips.append(dec(lp) != dec(r["ref"])); accs.append(r["labels"][dec(lp)] == r["gold"])
            acc_full = np.mean([r["labels"][dec(r["ref"])] == r["gold"] for r in rs])
            pol[grp] = {"custo_medio_relativo": float(np.mean([cost[l] for l in exit_l])),
                        "muda_vs_completo": float(np.mean(flips)), "acerto": float(np.mean(accs)),
                        "acerto_completo": float(acc_full), "camada_media_saida": float(np.mean(exit_l) + 1)}
        res["politica"][f"eps_{e}"] = {"limiares": taus, **pol}
    (out / "d2.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["politica"], indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------------------------------- D3
def cmd_estado(a):
    """Telepatia: as perguntas de um mesmo estado compartilham o prefixo. Prefill do prefixo comum (LCP dos
    tokens) uma vez e continuação de cada pergunta a partir do cache; confere contra a passada completa."""
    import torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    be, tpl = backend_native(), template()
    W = be._head_weight().float()
    cs = [c for sp in ("dev", "calib") for c in cases_for("telepatia", sp)]
    groups = defaultdict(list)
    for c in cs:
        groups[c.state].append(c)
    groups = [g for g in groups.values() if len(g) >= 2][: a.max_groups]

    def lcp(seqs):
        n = min(map(len, seqs)); k = 0
        while k < n and all(s[k] == seqs[0][k] for s in seqs):
            k += 1
        return k

    def sync():
        torch.cuda.synchronize()

    model = be.model

    def letters(o, cands):  # mesmo caminho nos dois lados: logits do modelo (LM head nativo), última posição
        return torch.log_softmax(o.logits[0, -1].float(), -1)[torch.tensor(cands, device=be.device)]

    def get_cache(o):
        c = getattr(o, "past_key_values", None)
        return c if c is not None else getattr(o, "cache_params", None)

    def step(input_ids, cache, pos):
        kw = dict(input_ids=input_ids, use_cache=True, cache_position=pos)
        try:
            return model(past_key_values=cache, **kw)
        except TypeError:
            return model(cache_params=cache, **kw)

    res, modes_ok = [], defaultdict(int)
    for gi, g in enumerate(groups):
        enc = [encode(be, tpl, c) for c in g]
        ids = [e[0] for e in enc]
        p = lcp(ids) - 1  # deixa ao menos 1 token de sufixo comum fora, para o sufixo nunca ser vazio
        # A: passada completa por pergunta (como no E1/E0 do grupo)
        sync(); t0 = time.perf_counter()
        ref = []
        with torch.inference_mode():
            for (x, cands, _) in enc:
                ref.append(letters(model(input_ids=torch.tensor([x], device=be.device)), cands))
        sync(); t_full = (time.perf_counter() - t0) * 1000
        # B: prefixo uma vez + continuação. Tenta a continuação em bloco; se não reproduzir, token a token.
        out_g = {"n_perguntas": len(g), "prefixo": p, "sufixo_medio": float(np.mean([len(x) - p for x in ids])),
                 "ms_completo_seq": t_full}
        for mode in ("bloco", "token"):
            try:
                sync(); t0 = time.perf_counter()
                with torch.inference_mode():
                    pre = model(input_ids=torch.tensor([ids[0][:p]], device=be.device), use_cache=True)
                    cache0 = get_cache(pre)
                    if cache0 is None:
                        raise RuntimeError("o modelo não devolveu cache com use_cache=True")
                    sync(); t_pre = (time.perf_counter() - t0) * 1000
                    diffs = []
                    for (x, cands, _), r in zip(enc, ref):
                        cache = copy.deepcopy(cache0)
                        suf = x[p:]
                        if mode == "bloco":
                            o = step(torch.tensor([suf], device=be.device), cache,
                                     torch.arange(p, len(x), device=be.device))
                        else:
                            for k, tkn in enumerate(suf):
                                o = step(torch.tensor([[tkn]], device=be.device), cache,
                                         torch.tensor([p + k], device=be.device))
                                nc = get_cache(o)
                                cache = nc if nc is not None else cache
                        diffs.append(float((letters(o, cands) - r).abs().max()))
                sync(); t_tot = (time.perf_counter() - t0) * 1000
                ok = max(diffs) <= a.tol
                out_g[mode] = {"ms_prefixo": t_pre, "ms_total": t_tot, "max_dif_logprob": max(diffs), "confere": ok}
                modes_ok[mode] += ok
            except Exception as e:  # registra e segue: o objetivo é saber o que o caminho HF suporta
                out_g[mode] = {"erro": f"{type(e).__name__}: {e}"[:300]}
        res.append(out_g)
        if gi % 20 == 0:
            print(json.dumps(out_g, ensure_ascii=False), flush=True)
    summ = {"grupos": len(res), "confere_por_modo": dict(modes_ok)}
    for mode in ("bloco", "token"):
        good = [r for r in res if r.get(mode, {}).get("confere")]
        if good:
            summ[mode] = {"aceleracao_mediana": float(np.median([r["ms_completo_seq"] / r[mode]["ms_total"] for r in good])),
                          "tokens_economizados": float(np.mean([1 - (r["prefixo"] + r["n_perguntas"] * r["sufixo_medio"])
                                                                 / (r["n_perguntas"] * (r["prefixo"] + r["sufixo_medio"])) for r in good]))}
    (out / "d3.json").write_text(json.dumps({"resumo": summ, "grupos": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------------------------------- D4
def cmd_conformal(a):
    """Learn-then-Test (Angelopoulos et al.): limiar com garantia P(risco entre aceitos <= α) >= 1-δ no calib
    (vistas), por teste de sequência fixa com p-valor binomial; aplicado fixo no dev, vistas e reservadas."""
    from scipy.stats import binom
    import ordem_abstencao as d1
    from lumi_decision import metrics
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"alfa": a.alpha, "delta": a.delta, "fontes": {}}
    for src in a.sources.split(","):
        calib = [r for r in d1.load(Path(a.runs) / f"{a.prefix}_{src}_calib_completo") if not r["held_out"]]
        dev = d1.load(Path(a.runs) / f"{a.prefix}_{src}_dev_completo")
        T = metrics.fit_temperature(calib)
        cT, dT = metrics.with_temperature(calib, T), metrics.with_temperature(dev, T)
        sig = {"S1": (d1.s1_items(cT), d1.s1_items(dT))}
        cc = [d1.case_signals(o) for o in d1.pair_cases(cT).values()]
        dd = [d1.case_signals(o) for o in d1.pair_cases(dT).values()]
        sig["S3"] = ([{**c, "conf": c["conf"]["S3"]} for c in cc], [{**c, "conf": c["conf"]["S3"]} for c in dd])
        sres = {}
        for s, (ci, di) in sig.items():
            conf = np.array([i["conf"] for i in ci]); err = np.array([i["pred"] != i["gold"] for i in ci], float)
            # S1 conta as duas ordens como itens; para a garantia, usar uma ordem por caso (independência)
            if s == "S1":
                keep = np.array([i["order"] == "canonical" for i in ci]); conf, err = conf[keep], err[keep]
            grid = np.unique(conf)[::-1]
            tau = None
            for t in grid:  # sequência fixa: do limiar mais alto ao mais baixo, para no primeiro que não rejeita
                acc = conf >= t; n, k = int(acc.sum()), int(err[acc].sum())
                if n == 0:
                    continue
                pval = binom.cdf(k, n, a.alpha)  # H0: risco > α
                if pval > a.delta:
                    break
                tau = float(t)
            g = {"limiar": tau}
            for grp, flag in (("vistas", False), ("reservadas", True)):
                its = [i for i in di if i["held_out"] == flag]
                if not its:
                    continue
                tab = d1.family_table(its, tau)
                g[grp] = {"risco_aceitos": d1.fam_mean(tab, "risco_aceitos"), "cobertura": d1.fam_mean(tab, "cobertura"),
                          "familias_que_violam_alfa": sorted(f for f, v in tab.items()
                                                             if v["risco_aceitos"] is not None and v["risco_aceitos"] > a.alpha),
                          "por_familia": tab}
            sres[s] = g
        res["fontes"][src] = sres
    (out / "d4.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({s: {k: {g: (v.get(g, {}).get("risco_aceitos"), v.get(g, {}).get("cobertura")) for g in ("vistas", "reservadas")}
                          for k, v in sr.items()} for s, sr in res["fontes"].items()}, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("saida"); s.add_argument("--out", required=True); s.add_argument("--sources", default="felhen,telepatia")
    s.add_argument("--batch", type=int, default=8); s.add_argument("--positions", type=int, default=16)
    s.add_argument("--train-max", type=int, default=4000); s.add_argument("--train-per-family", type=int, default=600)
    s.add_argument("--ridge", type=float, default=1e-3); s.add_argument("--tol", type=float, default=0.02)
    e = sub.add_parser("estado"); e.add_argument("--out", required=True); e.add_argument("--max-groups", type=int, default=300)
    e.add_argument("--tol", type=float, default=0.02)
    c = sub.add_parser("conformal"); c.add_argument("--runs", required=True); c.add_argument("--prefix", required=True)
    c.add_argument("--out", required=True); c.add_argument("--sources", default="felhen,telepatia,traces")
    c.add_argument("--alpha", type=float, default=0.10); c.add_argument("--delta", type=float, default=0.10)
    a = ap.parse_args()
    {"saida": cmd_saida, "estado": cmd_estado, "conformal": cmd_conformal}[a.cmd](a)


if __name__ == "__main__":
    main()
