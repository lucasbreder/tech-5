# Guardiã AI — apoio no atendimento à gestante

Aplicação do Tech Challenge Fase 5. O profissional informa sinais do atendimento e, se houver, um relato. A ferramenta devolve uma **leitura de apoio**: classe de risco de dois ou mais modelos, peso das variáveis, protocolos consultados e uma nota em linguagem clara.

A decisão continua com a pessoa responsável. A aplicação não diagnostica, não prescreve e não toma medida de segurança.

## Cenário

Triagem de **risco materno** (saúde da mulher), com uma camada de **escuta do relato** (segurança da mulher). O exemplo do edital — indicadores da gestação e interpretação de um relato no mesmo atendimento — é o fluxo desta tela.

## Dados

Conjunto público [UCI Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk) (`data/raw/`).

| Etapa | O que acontece |
| --- | --- |
| Unidades | Temperatura de °F para °C; glicemia de mmol/L para mg/dL |
| Rótulos | `low/mid/high risk` → `baixo` / `moderado` / `alto` |
| Qualidade | Sai frequência cardíaca incompatível (há registro de 7 bpm) |
| Vazamento | Duplicatas saem antes do split |
| Rótulo ambíguo | 35 vetores idênticos tinham riscos diferentes; as 71 linhas desses grupos saem por não haver dado clínico para desempatar |

O fluxo passa de 1.014 linhas brutas para 451 linhas sem duplicação exata e
380 linhas sem conflito de rótulo. Variáveis: idade, pressão sistólica,
pressão diastólica, glicemia, temperatura e frequência cardíaca.

## Modelos

Três classificadores são comparados com validação cruzada estratificada de
cinco folds apenas no conjunto de treino. A seleção usa o **recall médio da
classe alto**; empate desempata pela **precisão média da classe alto**. Depois,
o resultado final é medido uma vez no teste estratificado (20%, 76 linhas).

| Modelo | Accuracy | Recall alto | Precisão alto | F1 |
| --- | --- | --- | --- | --- |
| random forest | 0,829 | 0,950 | 0,950 | 0,832 |
| gradient boosting | 0,763 | 0,900 | 0,947 | 0,743 |
| regressão logística | 0,711 | 0,750 | 0,833 | 0,698 |

Na validação cruzada, a random forest teve recall alto médio de 0,793
(desvio 0,156) e foi escolhida. No teste final ela encontrou 95% dos casos
altos. O teste é pequeno; esse número não é estimativa clínica externa.

A tela separa a importância global do **efeito local** deste atendimento
(diferença de probabilidade ao trocar uma variável pela mediana do treino).
SHAP médio também é calculado na random forest. A classe moderado continua
mais fraca e deve ser tratada como limitação.

Isolation Forest não entra nessa tabela. Ele só marca combinação incomum em relação ao treino.

## Fluxo

```
Atendimento → classificadores → faixa de atenção e relato
           → protocolos (RAG) → nota de apoio → profissional
```

LangGraph encadeia esses passos. A busca nos protocolos é separada: uma consulta para os sinais clínicos e, quando há relato, outra para escuta e sigilo. Cada trecho mostra o arquivo de origem.

A nota sai de um modelo **local** (Ollama, padrão `llama3`), sem chave de API. O servidor precisa estar no ar: `ollama serve`.

Toda análise vai para `data/reports/audits/*.jsonl` (entrada, classe, fontes e nota).

## Como executar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# opcional: cp .env.example .env  (OLLAMA_MODEL, se quiser outro modelo local)

ollama serve                 # em outro terminal
ollama pull llama3           # apenas se ainda não estiver instalado
python main.py train
python main.py rag-ingest
python main.py demo    # http://localhost:8501
python main.py report  # relatorio/relatorio_tecnico.pdf
```

Um caso pela linha de comando (unidades da tela: °C e mg/dL):

```bash
python main.py predict \
  --features '{"age":34,"systolic_bp":148,"diastolic_bp":96,"blood_sugar":168,"body_temp":36.8,"heart_rate":92}' \
  --report "Queixa de dor de cabeça. Disse que tem medo de ir para casa porque houve ameaça."
```

Testes: `pytest`

Notebook de exploração e treino: `notebooks/01_eda_risco_materno.ipynb`

## Docker

A imagem treina os modelos e indexa os protocolos durante o build. O Compose
persiste apenas `data/reports/` e acessa o Ollama do host por
`host.docker.internal`.

```bash
ollama serve
docker compose up --build
```

No Linux, confirme que a versão do Docker Compose aceita
`host-gateway`. Para outro modelo: `OLLAMA_MODEL=nome docker compose up --build`.

## Entregáveis

- relatório fonte: `relatorio/RELATORIO_TECNICO.md`;
- PDF: `relatorio/relatorio_tecnico.pdf` (gerado por `python main.py report`);
- roteiro de demonstração: `VIDEO_SCRIPT.md`;
- notebook: `notebooks/01_eda_risco_materno.ipynb`;
- testes: `pytest -q`.

## Limites

- Conjunto público, pequeno depois de tirar duplicatas, sem idade gestacional, exame ou história obstétrica.
- Classe moderado é pouco recuperada.
- Termos do relato não confirmam violência; ausência de termo não confirma segurança.
- Faixas de atenção (por exemplo sistólica a partir de 140 mmHg) só destacam o número. Não são critério diagnóstico.
- Não usar prontuário real. A demonstração fica em dado público ou fictício.
