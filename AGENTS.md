# Guardiã AI — IA para Saúde e Segurança da Mulher (Fase 5)

> **Tech Challenge Fase 5.** Aplicação de IA que recebe informações de um atendimento e produz uma análise de apoio ao profissional, combinando **Dados + Machine Learning + LLM + RAG + aplicação**.
>
> A IA é **sistema de apoio à decisão** — nunca substitui o julgamento humano, não faz diagnóstico definitivo, não prescreve e não age automaticamente em situações de segurança.

## 1. Escopo do desafio (resumo do edital)

Escolher **um cenário principal** e construir a solução em torno dele. Dois contextos possíveis:

- **Saúde da mulher:** câncer de mama, risco materno / pré-eclâmpsia, complicações na gestação, saúde cardiovascular etc.
- **Segurança da mulher:** análise de relatos, organização de informações, identificação de situações que merecem atenção, consulta de protocolos.

**Fluxo obrigatório:**

```
Entrada dos dados → análise por ML → consulta de documentos (RAG)
                  → interpretação por LLM → apresentação ao profissional
```

### Requisitos que valem nota (checklist do edital)

| # | Requisito | Status |
|---|-----------|--------|
| 1 | Conjunto de dados relacionado ao problema (preferencialmente público/anonimizado/sintético) | ⬜ |
| 2 | Exploração + preparação dos dados | ⬜ |
| 3 | Treinar **≥ 2 modelos** de Machine Learning | ⬜ |
| 4 | Comparar com métricas adequadas (accuracy, precision, recall, F1) e **justificar** a escolha + impacto dos erros | ⬜ |
| 5 | Interpretabilidade: **feature importance / SHAP** (ou equivalente) | ⬜ |
| 6 | Integração com uma **LLM** (interpretar resultados, resumir relatos, explicar) | ⬜ |
| 7 | **RAG** com base de conhecimento (protocolos, cartilhas, docs públicos) + citar o documento usado | ⬜ |
| 8 | Orquestrar o fluxo com **LangChain** ou **LangGraph** (fluxo simples basta; agentes = diferencial) | ⬜ |
| 9 | Aplicação executável: **Streamlit / Gradio / API / web** | ⬜ |
| 10 | **Dockerfile** (execução containerizada) | ⬜ |
| 11 | **Registro/log das análises** (auditoria, explicabilidade) | ⬜ |
| 12 | Repositório Git: código + notebooks/scripts de treino + instruções + README | ⬜ |
| 13 | **Relatório técnico em PDF** (problema, dados, preparação, modelos, métricas, LLM, RAG, limitações) | ⬜ |
| 14 | **Vídeo de até 15 min** (YouTube/Vimeo, público ou não listado) com jornada completa | ⬜ |

### Cenário recomendado para o grupo

**Triagem de risco materno na gestação** (saúde) com camada de segurança (interpretação de relatos de violência doméstica):

- ML tabular clássico →数据集 público (ex.: **Maternal Health Risk / LIMA-DA / datasets Kaggle de risco gestacional**, sinais vitais + idade + histórico).
- Permite reaproveitar do tech-4 a lógica de **anomalias em sinais vitais** (Isolation Forest / Z-Score) como um dos modelos.
- RAG sobre protocolos pré-natal públicos (Ministério da Saúde / cartilhas SUS).

## 2. Decisões de arquitetura

```
                      +---------------------------+
                      |   Entrada do Atendimento  |
                      |  (dados tabulares + relato)|
                      +-------------+-------------+
                                    |
                 +------------------+------------------+
                 |                                     |
                 v                                     v
       +---------+---------+                  +--------+---------+
       |  ML Engine        |                  |  relato textual  |
       |  2+ modelos       |                  |                  |
       |  (RiskClassifier  |                  |                  |
       |   + AnomalyDetect)|                  |                  |
       +---------+---------+                  |                  |
                 |  + feature importance/SHAP |                  |
                 v                             v                  |
       +---------+-----------------------------+---------+       |
       |        LangGraph / LangChain Orchestrator       |◄──────┘
       |  dados → ML → retrieve (RAG) → LLM explain      |
       +----+-----------------------------+--------------+
            |                             |
            v                             v
   +--------+--------+            +-------+--------+
   |  Vector Store   |            |  LLM (OpenAI / |
   |  (protocolos /  |            |  Azure OpenAI) |
   |   cartilhas)    |            +-------+--------+
   +-----------------+                    |
                                          v
                              +-----------+-----------+
                              |  Apresentação ao      |
                              |  profissional         |
                              |  (Streamlit)          |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------+-----------+
                              |  Audit Log (JSONL)    |
                              |  análise + docs usados|
                              +-----------------------+
```

## 3. Estrutura do projeto

> Layout **herdado/refinado do tech-4** (`src/` modular, config centralizada via Pydantic Settings, Loguru, Typer CLI, Streamlit), adaptado para ML tabular + LLM + RAG.

```
tech-5/
├── main.py                         # CLI (Typer): train, predict, rag-ingest, demo, explain
├── requirements.txt                # Dependências
├── Dockerfile                      # Execução containerizada (requisito #10)
├── docker-compose.yml              # (opcional) app + vector store
├── .env.example                    # Template de variáveis (chaves OpenAI/Azure etc.)
├── .gitignore
├── AGENTS.md                       # Este arquivo (arquitetura + plano)
├── README.md                       # Problema, arquitetura, resultados, como rodar
├── src/
│   ├── __init__.py
│   ├── config.py                   # Pydantic Settings (reaproveitar padrão tech-4)
│   ├── app.py                      # Interface Streamlit (requisito #9)
│   ├── data/
│   │   ├── loader.py               # Carrega dataset tabular (CSV/parquet)
│   │   └── preprocessing.py        # Limpeza, encoding, escala, split
│   ├── ml/
│   │   ├── train.py                # Treina os 2+ modelos (requisito #3)
│   │   ├── models.py               # ex.: RandomForest, GradientBoosting, LogisticReg
│   │   ├── evaluate.py             # precision/recall/F1/accuracy + matriz + comparação (#4)
│   │   ├── explain.py              # SHAP / feature importance (#5)
│   │   └── anomaly.py              # Isolation Forest / Z-Score (reuso do tech-4)
│   ├── rag/
│   │   ├── ingest.py               # Chunking + embeddings de protocolos/cartilhas (#7)
│   │   ├── store.py                # Vector store (Chroma/FAISS)
│   │   └── retriever.py            # Busca por similaridade + retorna fonte/documento
│   ├── llm/
│   │   ├── client.py               # Wrapper OpenAI / Azure OpenAI (reusa config tech-4)
│   │   ├── prompts.py              # Prompt de explicação (anti-recomendação definitiva)
│   │   └── chains.py               # Cadeias LangChain (#6)
│   ├── graph/
│   │   └── workflow.py             # LangGraph: dados → ML → retrieve → LLM → resultado (#8)
│   ├── services/
│   │   └── audit_log.py            # Registro JSONL das análises (#11) — padrão de alert_service tech-4
│   └── utils/
│       └── report_generator.py     # JSON + Markdown (reuso tech-4) → base p/ PDF (#13)
├── data/
│   ├── raw/                        # Dataset bruto
│   ├── processed/                  # Após preparação
│   ├── knowledge/                  # Protocolos/cartilhas p/ RAG (PDFs/MD públicos)
│   └── reports/
│       └── audits/                 # Logs de auditoria (JSONL)
├── models/                         # Modelos treinados (.pkl / joblib)
├── notebooks/                      # EDA + experimentos de ML (entrega)
├── tests/                          # Testes (pytest)
└── edital/
    └── POSTECH - HACKA IADT - Secretaria - Fase 5.md
```

## 4. O que reaproveitar do tech-4 (e o que NÃO levar)

O tech-4 foi **monitoramento multimodal (vídeo/áudio/vitais)**. O tech-5 é **ML tabular + LLM + RAG** — foco diferente. Reaproveitar **infra e padrões de código**, não os pipelines de visão computacional.

### ✅ Reaproveitar (adaptar)

| Item do tech-4 | Uso no tech-5 |
|----------------|---------------|
| `src/config.py` — Pydantic Settings + `resolve_device()` + caminhos de diretório | Mesma base de config; trocar chaves Azure por OpenAI/Azure OpenAI + vector store |
| `.env.example` (chaves Azure/OpenAI) | Manter `OPENAI_API_KEY` / `AZURE_OPENAI_*`; descartar Speech/Vision/Storage |
| `services/alert_service.py` (SMTP, logging, JSON) | Modelo para `services/audit_log.py` (registro das análises — requisito #11) |
| `utils/report_generator.py` (JSON + Markdown/Jinja2) | Base para gerar o relatório técnico em PDF (#13) |
| `agents/anomaly_agent.py` (Isolation Forest + Z-Score + faixas de gestante) | Vira `ml/anomaly.py` — **um dos 2+ modelos** (detecção de risco materno) |
| `app.py` Streamlit (abas, barras de progresso) | Esqueleto da UI de triagem |
| `main.py` Typer | CLI (train / predict / rag-ingest / demo) |
| Loguru, Pandas, Scikit-learn, Pydantic v2 | Mesmas libs base |
| Padrão `VIDEO_SCRIPT.md` (roteiro) | Roteiro do vídeo de 15 min (#14) |

### ⬜ Não reaproveitar (fora do escopo Fase 5)

- YOLOv8 / OpenCV / MoviePy (visão computacional) — só se o grupo optar por áudio/imagem/vídeo como **evolução opcional**.
- Whisper / Wav2Vec2 / Azure Speech & Vision — idem (opcional).
- Datasets de imagens (emotions/blood/surgery_instruments) — irrelevantes para ML tabular de saúde.

### ⭐ Novidades a estudar/implementar (não existiam no tech-4)

- **LangChain / LangGraph** (#8) — orquestração do fluxo.
- **RAG** (#7): embeddings + vector store (Chroma ou FAISS) + retriever com citação da fonte.
- **SHAP / feature importance** (#5).
- **≥2 modelos ML** comparados com métricas justificadas (#3, #4).
- **Dockerfile** (#10) — tech-4 **não tinha** Docker; criar do zero.

## 5. Bibliotecas sugeridas (requirements.txt)

```
# Core ML
scikit-learn>=1.5
pandas>=2.2
numpy>=1.26
joblib>=1.4
shap>=0.46            # interpretabilidade (#5)
imbalanced-learn>=0.12# se houver desbalanceio de classes

# LLM + RAG
langchain>=0.3
langgraph>=0.2        # (#8)
langchain-openai>=0.2
langchain-community>=0.3
chromadb>=0.5         # ou faiss-cpu + langchain
openai>=1.40          # ou azure-openai

# App / CLI / config
streamlit>=1.36
typer>=0.12
pydantic-settings>=2.3
python-dotenv>=1.0
loguru>=0.7
jinja2>=3.1

# Relatório PDF
md2pdf>=1.0           # ou weasyprint / reportlab

# Testes
pytest>=8.0
```

## 6. Plano de execução (sugestão de ordem)

1. **Definir cenário** (ex.: risco materno) + escolher dataset público → salvar em `data/raw/`.
2. **EDA + preprocessing** (notebook + `src/data/`).
3. **Treinar 2+ modelos** → `src/ml/` → comparar métricas e **justificar** recall vs precision (falso negativo em risco materno é crítico).
4. **Explicabilidade** com SHAP.
5. **Base de conhecimento RAG**: baixar protocolos/cartilhas públicas → `data/knowledge/` → `src/rag/ingest.py`.
6. **LLM**: prompt de explicação responsável (sem recomendação definitiva) → `src/llm/`.
7. **Orquestrar** tudo em LangGraph → `src/graph/workflow.py`.
8. **UI** Streamlit com a jornada completa + exibição da fonte do RAG.
9. **Audit log** de cada análise.
10. **Dockerfile** + testar `docker build`/`docker run`.
11. **README** + **relatório PDF** + **vídeo (≤15 min)**.

## 7. Como executar (alvo)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # preencher OPENAI_API_KEY (ou Azure OpenAI)

# Ingerir base de conhecimento (RAG)
python main.py rag-ingest data/knowledge/

# Treinar modelos + gerar métricas/comparação + SHAP
python main.py train --dataset data/raw/maters.csv

# Rodar a aplicação (jornada completa)
python main.py demo           # abre Streamlit

# Docker
docker build -t guardia-ai .
docker run -p 8501:8501 --env-file .env guardia-ai
```

## 8. Diretrizes de segurança e responsabilidade (obrigatórias)

- Usar **apenas dados públicos, anonimizados ou sintéticos** — nunca dados reais de pacientes.
- UI e respostas da LLM devem deixar explícito: **apoio ao profissional, não substitui decisão humana**.
- LLM **não** deve emitir recomendação definitiva / diagnóstico / prescrição quando faltar informação → validar no prompt e sinalizar na tela.
- Toda análise deve ser **registrada** (audit log) com: dados de entrada, resultado do modelo, documentos RAG usados e resposta da LLM (rastreabilidade / explicabilidade).
- Ao citar RAG, **sempre indicar a fonte/documento** utilizado na resposta.

## 9. Convenções

- Type hints + docstrings (padrão tech-4).
- Config sempre via env/Pydantic — **nunca** commitar chaves (`.env` no `.gitignore`).
- `models/`, `data/raw/`, `data/processed/` e artefatos grandes fora do Git (via `.gitignore`).
- Commits no padrão do tech-4: `feat:` / `fix:` / `docs:` em minúsculas.
- Vídeo/áudio/imagem/vídeo e agentes autônomos = **diferencial opcional**, nunca pré-requisito.
