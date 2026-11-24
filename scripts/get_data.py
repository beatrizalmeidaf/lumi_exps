"""Baixa os dados dos experimentos do Lumi para $LUMI_DATA (padrao /raid/$USER/lumi_data).

Avaliacao (multipla escolha):
  mmlu_pt.parquet   MMMLU PT-BR (openai/MMMLU), traducao humana do MMLU de teste
  mmlu_en.parquet   MMLU original (cais/mmlu, all/test), mesma ordem -> pares PT/EN item a item
  enem.parquet, bluex.parquet, oab.parquet   provas brasileiras (Open PT LLM Leaderboard)
Texto para TV por token e calibracao:
  flores_{por,eng}.parquet   FLORES+ devtest, frases paralelas
  web_pt.parquet             FineWeb-2 por_Latn (shard de teste)
  web_en.parquet             C4 validation (en), shard local
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(os.environ.get("LUMI_DATA", f"/raid/{os.environ['USER']}/lumi_data"))
OUT.mkdir(parents=True, exist_ok=True)


def get(repo: str, fn: str) -> str:
    return hf_hub_download(repo, fn, repo_type="dataset", cache_dir=str(OUT / "_hf"))


def mmlu() -> None:
    pt = pd.read_csv(get("openai/MMMLU", "test/mmlu_PT-BR.csv"))
    en = pd.read_parquet(get("cais/mmlu", "all/test-00000-of-00001.parquet"))
    assert len(pt) == len(en), (len(pt), len(en))
    letters = "ABCD"
    en_ans = en["answer"].map(lambda i: letters[i])
    agree = (en_ans.values == pt["Answer"].values).mean()
    subj_agree = (en["subject"].values == pt["Subject"].values).mean()
    print(f"MMLU pares: {len(pt)} itens, gabarito igual em {agree:.4f}, materia igual em {subj_agree:.4f}")
    assert agree > 0.99 and subj_agree > 0.99, "MMMLU e MMLU nao estao na mesma ordem"
    pd.DataFrame({"id": range(len(pt)), "subject": pt["Subject"], "question": pt["Question"],
                  "choices": pt[["A", "B", "C", "D"]].values.tolist(), "answer": pt["Answer"]}
                 ).to_parquet(OUT / "mmlu_pt.parquet")
    pd.DataFrame({"id": range(len(en)), "subject": en["subject"], "question": en["question"],
                  "choices": [list(c) for c in en["choices"]], "answer": en_ans}
                 ).to_parquet(OUT / "mmlu_en.parquet")


def br_exams() -> None:
    for name, repo in [("enem", "eduagarcia/enem_challenge"),
                       ("bluex", "eduagarcia-temp/BLUEX_without_images"),
                       ("oab", "eduagarcia/oab_exams")]:
        df = pd.read_parquet(get(repo, "data/train-00000-of-00001.parquet"))
        print(name, len(df), list(df.columns))
        df.to_parquet(OUT / f"{name}_raw.parquet")


def flores() -> None:
    for lang in ["por_Latn", "eng_Latn"]:
        try:
            df = pd.read_parquet(get("openlanguagedata/flores_plus", f"devtest/{lang}.parquet"))
        except Exception as e:  # noqa: BLE001
            print(f"[flores] falhou {lang}: {e}")
            return
        df[["id", "text"]].to_parquet(OUT / f"flores_{lang[:3]}.parquet")
        print("flores", lang, len(df))


def web() -> None:
    pt = pd.read_parquet(get("HuggingFaceFW/fineweb-2", "data/por_Latn/test/000_00000.parquet"),
                         columns=["text"])
    pt.sample(n=min(4000, len(pt)), random_state=0).to_parquet(OUT / "web_pt.parquet")
    en = pd.read_json(os.environ["C4_VALIDATION_FILE"], lines=True, nrows=20000)[["text"]]
    en.sample(n=4000, random_state=0).to_parquet(OUT / "web_en.parquet")
    print("web", len(pt), len(en))


if __name__ == "__main__":
    mmlu()
    br_exams()
    flores()
    web()
