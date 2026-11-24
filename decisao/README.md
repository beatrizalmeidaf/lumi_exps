# D1: o desacordo entre ordens prevê erro melhor que a probabilidade calibrada?

Protocolo registrado em 2026-11-24, **antes** de qualquer execução. Usa o E1 do `lumi-decision` (Lumi-4B
congelado, readout pelas letras, ordens canônica e invertida, dev e calib de Felhen, Typed Decisions PT e
Traces), rodado por nós no `h100n3` com o código e as revisões fixadas do grupo. Nenhum treino.

## Pergunta

Hoje a abstenção usa a probabilidade máxima após temperatura, com limiar escolhido no `calib` (só famílias
vistas) para 10% de erro. Nas famílias reservadas ela não transfere: o Lumi congelado responde 7–17% dos casos
com 23–31% de erro (README do grupo, tabela "Abstenção"). Um caso cuja escolha muda ao inverter as opções é
duvidoso por construção, e esse sinal não depende de quão calibrada a família está.

## Sinais (um caso = as duas ordens do mesmo `case_id`)

| | Sinal | Decisão | Confiança | Passadas |
|---|---|---|---|---|
| S1 | **referência do grupo**: cada ordem sozinha | argmax da ordem | max p_T da ordem | 1 |
| S2 | média das ordens | argmax de p̄ = (p_T,can + p_T,inv)/2 | max p̄ | 2 |
| S3 | **média com veto por desacordo** | argmax de p̄ | max p̄ se as duas ordens concordam; 0 se discordam | 2 |
| S4 | pior caso entre ordens | argmax de p̄ | min das duas ordens da prob. da opção escolhida | 2 |

p_T é a distribuição sobre as opções com a temperatura T ajustada no `calib` (`metrics.fit_temperature`, linhas
das duas ordens). S1 é avaliado em cada ordem e as métricas são a média das duas, como na métrica principal do
grupo.

## Medidas (no dev; famílias vistas e reservadas separadas, média com peso igual por família)

* **Principal, transferência:** com o limiar de cada sinal escolhido no `calib` para risco-alvo 10%
  (`threshold_for_risk`, mínimo de 20 aceitos) e aplicado fixo no dev, o **risco entre os aceitos nas
  reservadas**. Melhora se o risco realizado de S3 ficar abaixo do de S1, com IC 95% da diferença inteiro abaixo
  de zero (bootstrap de grupos de texto/estado dentro de cada família, 2000 reamostragens, mesmo esquema do
  `family_mean_bootstrap`).
* **Secundárias:** cobertura com o limiar fixo; AURC (sem limiar); risco a 50% de cobertura; mesmas medidas nas
  vistas; para S1, a fração das decisões aceitas que mudam com a ordem.
* **Custo:** S2–S4 custam duas passadas. Só valem a pena se o ganho em risco aceito for maior do que o de dar a
  S1 o mesmo orçamento, o que aqui significa comparar S3 com S2 (mesma decisão, mesmo custo, só muda o veto).

## Leitura dos resultados

* S3 < S1 nas reservadas com IC < 0 → o desacordo é um sinal de abstenção que transfere; segue para a parte de
  compressão (flip nas decisões aceitas, FP8/W4) e serviço.
* S3 ≈ S2 → o ganho vem da média das ordens, não do veto.
* Nada melhora → a abstenção não transfere nem com esse sinal; registrar como resultado negativo.

Limites: uma semente (modelo congelado, sem variação de treino); só duas ordens; Traces é só diagnóstico, como
no grupo.

---

# D2: saída antecipada para decisões (registrado em 2026-11-24, antes de rodar)

**Pergunta:** a decisão do Lumi congelado (distribuição sobre as letras) já está pronta antes da camada 42? Se
estiver, ler a decisão numa camada intermediária corta o custo do prefill de cada decisão.

* **Leituras por camada:** logit lens (norma final + LM head) e tuned lens afim por camada (ridge em forma
  fechada, h_l → h_42 antes da norma), ajustado nas últimas 16 posições de até 4.000 casos de **treino**
  (Felhen + Typed Decisions PT, famílias vistas; o modelo congelado nunca treinou neles e eles não entram na
  avaliação). Ordem canônica.
* **Sanidade (derruba o job):** a camada 42 pelo logit lens reproduz a leitura oficial (|Δlog-prob| ≤ 0,02).
* **Política:** sai na primeira camada l ≥ 8 cuja probabilidade máxima (tuned lens) passa um limiar τ_l
  escolhido no `calib` para que no máximo ε (1% e 2%) dos casos que saem em l mudem de decisão em relação ao
  modelo completo. Avaliada no `dev`, vistas e reservadas.
* **Medidas:** custo relativo médio (tempo de GPU acumulado por camada, medido com eventos CUDA), taxa de
  decisões que mudam em relação ao modelo completo, acerto com e sem saída antecipada.
* **Leitura:** vale a pena se o custo médio cair ≥ 20% com mudança ≤ ε também nas reservadas.

# D3: reaproveitar o estado do contexto entre perguntas (registrado em 2026-11-24, antes de rodar)

**Pergunta:** no Typed Decisions PT as perguntas de um mesmo estado compartilham o prefixo do prompt. Fazer o
prefill do prefixo comum uma vez (estado do Mamba + KV das 4 camadas de atenção) e continuar cada pergunta a
partir do cache dá as mesmas decisões, mais rápido?

* Até 300 estados de dev + calib com ≥ 2 perguntas; prefixo = maior prefixo comum de tokens menos 1.
* **Correção:** log-probs das letras iguais às da passada completa (|Δ| ≤ 0,02, mesmo LM head dos dois lados).
  Dois modos de continuação: em bloco (o sufixo inteiro numa chamada) e token a token. O job registra qual modo
  o caminho HF suporta corretamente.
* **Medidas:** aceleração mediana por estado (completo ÷ prefixo + continuações) e fração de tokens
  economizados. Só os modos que conferem contam.

# D4: abstenção com garantia (registrado em 2026-11-24, antes de rodar)

**Pergunta:** um limiar com garantia estatística no `calib` (Learn-then-Test, sequência fixa com p-valor
binomial: P(risco entre aceitos ≤ α) ≥ 1−δ, α = δ = 0,10) mantém o risco nas famílias vistas e quanto ele é
violado nas reservadas, para S1 (referência) e S3 (veto por desacordo)?

* Para a garantia, S1 usa uma ordem por caso (canônica), para os itens serem independentes.
* **Medidas:** risco entre aceitos e cobertura no dev (vistas e reservadas, média por família) e as famílias
  que violam α. A violação nas reservadas mede diretamente quanto a calibração não transfere.
