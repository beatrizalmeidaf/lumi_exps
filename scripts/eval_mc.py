"""Avaliacao de multipla escolha com vLLM, thinking desligado (como o model card do Lumi exige).

Prompt de chat zero-shot; a resposta e a letra de maior logprob no 1o token gerado (top-20).
Se nenhuma letra valida aparece no top-20, o item conta como erro (coluna `miss`).

Tarefas:
  mmlu_pt / mmlu_en   MMMLU PT-BR e MMLU original: os MESMOS 14 042 itens nos dois idiomas
  enem, bluex, oab    provas brasileiras (sem itens anulados)

Saida: $LUMI_RESULTS/e1/eval/<tag>.parquet  (uma linha por item; probabilidades das letras)
Uso:  python lumi/scripts/eval_mc.py --model <repo ou dir> --tag lumi4b__bf16 [--tasks ...] [--limit N]
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ["LUMI_DATA"])
OUT = Path(os.environ["LUMI_RESULTS"]) / "e1" / "eval"

INSTR = {
    "pt": "Responda à questão de múltipla escolha a seguir. Responda apenas com a letra da alternativa correta.",
    "en": "Answer the following multiple-choice question. Reply with only the letter of the correct option.",
}


def load_task(task: str) -> pd.DataFrame:
    """Colunas: id, subject, question, labels (lista), choices (lista), answer, lang."""
    if task in ("mmlu_pt", "mmlu_en"):
        df = pd.read_parquet(DATA / f"{task}.parquet")
        df["labels"] = [list("ABCD")] * len(df)
        df["choices"] = df["choices"].map(list)
        df["lang"] = task[-2:]
        return df
    raw = pd.read_parquet(DATA / f"{task}_raw.parquet")
    if "nullified" in raw:
        raw = raw[~raw["nullified"].fillna(False).astype(bool)]
    return pd.DataFrame({
        "id": raw["id"].astype(str).values,
        "subject": raw.get("exam_id", pd.Series([task] * len(raw))).astype(str).values,
        "question": raw["question"].values,
        "labels": [list(c["label"]) for c in raw["choices"]],
        "choices": [list(c["text"]) for c in raw["choices"]],
        "answer": raw["answerKey"].values,
        "lang": "pt",
    })


def user_msg(r) -> str:
    opts = "\n".join(f"{l}) {str(c).strip()}" for l, c in zip(r.labels, r.choices))
    return f"{INSTR[r.lang]}\n\n{str(r.question).strip()}\n\n{opts}"


def top_vllm(args, msgs) -> list[list[tuple[str, float]]]:
    """Top-20 do 1o token gerado: [(texto do token, logprob)] por item."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              trust_remote_code=False, quantization=args.quantization, kv_cache_dtype=args.kv_cache_dtype,
              seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for m in msgs]
    res = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    return [[(tok.decode([tid]), v.logprob) for tid, v in o.outputs[0].logprobs[0].items()] for o in res]


def top_api(url: str, msgs, workers: int) -> list[list[tuple[str, float]]]:
    """O mesmo pelo /v1/chat/completions (llama-server com --jinja)."""
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def one(m):
        body = {"messages": m, "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=600) as f:
                    j = json.load(f)
                c = j["choices"][0]["logprobs"]["content"][0]
                return [(t["token"], t["logprob"]) for t in c["top_logprobs"]]
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, msgs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tasks", default="mmlu_pt,mmlu_en,enem,bluex,oab")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--quantization", default=None, help="quantizacao real do vLLM (ex.: fp8)")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--api", default=None, help="URL base de um servidor OpenAI (ex.: llama-server)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mmlu-n", type=int, default=0, help="subamostra pareada do MMLU (0 = todos)")
    args = ap.parse_args()

    out = OUT / f"{args.tag}.parquet"
    if out.exists():
        print(f"[skip] {out}")
        return
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    for t in args.tasks.split(","):
        d = load_task(t)
        if args.mmlu_n and t.startswith("mmlu"):
            d = d.sample(n=args.mmlu_n, random_state=0)   # mesmo sorteio nos dois idiomas
        if args.limit:
            d = d.head(args.limit)
        d["task"] = t
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["id"] = df["id"].astype(str)
    msgs = [[{"role": "user", "content": user_msg(r)}] for r in df.itertuples()]

    t0 = time.time()
    tops = top_api(args.api, msgs, args.workers) if args.api else top_vllm(args, msgs)
    secs = time.time() - t0

    preds, miss, probs = [], [], []
    for r, top in zip(df.itertuples(), tops):
        p = {L: 0.0 for L in r.labels}
        for s, logprob in top:
            L = s.strip()
            if L in p:
                p[L] += math.exp(logprob)
        tot = sum(p.values())
        miss.append(tot == 0.0)
        preds.append(max(p, key=p.get) if tot > 0 else "")
        probs.append([p[L] / tot if tot > 0 else 0.0 for L in r.labels])
    df["pred"], df["miss"], df["probs"] = preds, miss, probs
    df["correct"] = df["pred"] == df["answer"]
    df["model"], df["tag"] = args.model, args.tag
    df.drop(columns=["question", "choices"]).to_parquet(out)
    summ = df.groupby("task").agg(acc=("correct", "mean"), miss=("miss", "mean"), n=("correct", "size"))
    print(summ.to_string())
    print(f"[ok] {out} ({secs:.0f} s de geracao, {len(df)} itens)")


if __name__ == "__main__":
    main()
