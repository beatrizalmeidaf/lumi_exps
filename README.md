# Lumi-4B: compressão, qualidade por idioma e velocidade de serviço

Experimentos sobre o [CEIA Lumi-4B](https://huggingface.co/ceia-research/lumi-4b) (NemotronH híbrido
Mamba2 + atenção, CPT+KD+SFT em PT-BR sobre o Nemotron-3-Nano-4B). Os relatórios ficam em `relatorios/`;
a bancada de runtimes da RTX 3090/3060 (sparkbench, Rafael) é a referência de protocolo para o E2.

Pré-registro escrito em 2026-11-15, antes de qualquer resultado com pesos comprimidos (só o BF16 do Lumi
tinha sido avaliado).

## Arquitetura (o que importa para comprimir)

42 camadas: 21 Mamba2 (`in_proj` 17504×3136, `out_proj` 3136×7680), 17 MLP ReLU² (`up/down_proj`,
12544), 4 atenção GQA (8 cabeças KV). 3,15 B dos 3,97 B parâmetros estão nas 92 `Linear` das camadas.
Consequências: o KV cache é pequeno (4 camadas, 16 KiB/token em BF16, contra 144 KiB/token do Qwen3-4B)
e o estado do Mamba é fixo por sequência (~41 MB), então a memória por usuário quase não cresce com o
contexto.

## E1. A quantização apaga primeiro o português aprendido no CPT?

Lumi e Nemotron base, cada um em BF16 e em 9 compressões simuladas (pesos desquantizados em BF16, mesmo
pipeline do GLOD): RTN W4/W3 g128; GPTQ W4/W3 g128 com calibração em inglês (C4) ou português
(FineWeb-2 por_Latn); e RTN W3 aplicado só aos blocos Mamba, só às MLP ou só à atenção.

Medidas:
* acurácia de múltipla escolha, thinking desligado, letra de maior probabilidade no 1º token:
  MMLU (EN) e MMMLU (PT), os mesmos 14 042 itens nos dois idiomas; ENEM, BLUEX e OAB;
* TV, KL e taxa de troca do argmax por token contra o pai BF16 (estilo GLOD), em texto web PT/EN e nos
  prompts do MMLU PT/EN.

Hipóteses (testes por bootstrap pareado por item, 10 000 reamostragens, IC 95%):
* **H1 (fragilidade do CPT).** O ganho do Lumi sobre o Nemotron em PT (MMMLU + provas) encolhe com a
  quantização mais do que o ganho em EN. Critério: em W3 (RTN e GPTQ), a queda do ganho em PT é maior que
  a do ganho em EN, com IC que exclui zero em pelo menos 2 das 3 configurações W3.
* **H2 (idioma, dentro do Lumi).** Nos itens pareados do MMLU, o Lumi perde mais acurácia em PT do que em
  EN ao ser quantizado. Critério: diferença das quedas (PT − EN) > 0 com IC que exclui zero em W3.
* **H3 (calibração).** GPTQ calibrado em PT reduz a perda em PT em relação ao calibrado em EN, no Lumi.
  Critério: queda em PT com cal-pt < com cal-en em W3, IC que exclui zero.
* **H4 (GLOD).** A TV por token em PT ordena as perdas de acurácia em PT entre as configurações
  (Spearman ≥ 0,7 por modelo).
* Exploratório, sem direção pré-registrada: qual tipo de bloco (Mamba, MLP, atenção) concentra a
  sensibilidade, normalizado por número de parâmetros.

## E2. Lumi em formatos reais: qualidade e velocidade

O mesmo modelo em precisões diferentes, com kernels reais (a seção §6 que está pendente no relatório da
RTX 3090):
* vLLM: BF16, FP8 W8A8 dinâmico, FP8 + KV FP8, W4A16 GPTQ (llm-compressor, Marlin) calibrado em PT e em EN;
* llama.cpp: GGUF BF16, Q8_0, Q4_K_M sem imatrix, com imatrix EN e com imatrix PT.

Qualidade: mesma avaliação do E1 (vLLM direto; GGUF via llama-server), mais KLD do llama.cpp contra o BF16
em texto PT e EN. Velocidade: protocolo da sparkbench (512→256, `ignore_eos`, carga fechada 1–128
usuários), métricas iguais (tok/s com 1 usuário, vazão máxima, TTFT P99 c=16, TPOT P50 c=64, capacidade
no SLO B, tok/J, fração do teto de banda).

Expectativa (não é hipótese testada): o W4A16 deve ser o melhor formato com 1 usuário (lê ~3,5× menos
bytes) e o FP8 o melhor sob carga no H100; a imatrix em PT deve reduzir a KLD em PT do Q4_K_M.

## Estrutura

```
env.sh                    ambiente (herda ../env.sh; pesos em /raid/$USER/hf_extra, dados em /raid/$USER/lumi_data)
scripts/get_data.py       dados de avaliação, texto web e pares MMLU/MMMLU
scripts/quantize.py       compressões simuladas (E1)
scripts/eval_mc.py        múltipla escolha (vLLM ou servidor OpenAI)
scripts/tv_lang.py        TV/KL/troca por token contra o pai, por idioma
scripts/e1.sbatch         pipeline do E1
scripts/make_w4a16.py     W4A16 real com llm-compressor (venv próprio em /raid/$USER/lumi_tools)
scripts/bench_serve.sh    bancada de serviço com o protocolo da sparkbench
scripts/e2.sbatch         pipeline do E2
scripts/analyze.py        tabelas, testes e figuras dos relatórios
results/e1, results/e2    resultados (parquet/json/csv)
relatorios/               relatórios (HTML + PDF)
```
