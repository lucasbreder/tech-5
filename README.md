# 🛡️ Guardiã AI — Inteligência Artificial para Saúde e Segurança da Mulher

**Tech Challenge — Fase 5**

Aplicação de IA que recebe informações de um atendimento e produz uma análise
de **apoio ao profissional**, combinando **Dados + Machine Learning + LLM + RAG**.

> ⚠️ A Guardiã AI **não** faz diagnóstico definitivo, **não** prescreve e **não**
> toma decisões automáticas. A avaliação final é sempre humana.

## Arquitetura

```
Entrada dos dados → ML (2+ modelos + SHAP) → RAG (protocolos) → LLM (explicação) → Profissional
                                              └────────── LangGraph / LangChain ──────────┘
```

Detalhes em [`AGENTS.md`](AGENTS.md). Estrutura do código (reaproveitando padrões
do tech-4):

```
src/
├── config.py            # Pydantic Settings (.env)
├── app.py               # Streamlit (jornada completa)
├── data/                # loader + preprocessing
├── ml/                  # train, models, evaluate, explain (SHAP), anomaly
├── rag/                 # ingest, store (Chroma), retriever (cita fonte)
├── llm/                 # client, prompts (responsáveis), chains
├── graph/workflow.py    # orquestração LangGraph/LangChain
├── services/audit_log   # registro das análises (JSONL)
└── utils/report_generator  # Markdown → base p/ PDF
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # preencha OPENAI_API_KEY (ou Azure OpenAI)
```

## Uso

```bash
# 1. Coloque o dataset em data/raw/ e os protocolos em data/knowledge/ (.md/.txt)
python main.py rag-ingest           # indexa a base de conhecimento (RAG)
python main.py train                # treina os modelos + comparação de métricas
python main.py predict --features '{"age":28,"systolic_bp":138}' --report "..."
python main.py demo                 # interface Streamlit (http://localhost:8501)
```

## Docker

```bash
docker build -t guardia-ai .
docker run -p 8501:8501 --env-file .env guardia-ai
# ou: docker compose up --build
```

## Testes

```bash
pytest
```

## Entregáveis do edital (checklist)

Ver tabela completa em [`AGENTS.md`](AGENTS.md#1-escopo-do-desafio-resumo-do-edital).

- [x] Esqueleto: dados, ML (2+ modelos), SHAP, RAG, LLM, LangGraph, Streamlit, Docker, auditoria
- [ ] Dataset escolhido + EDA em `notebooks/`
- [ ] Relatório técnico PDF (`relatorio/`)
- [ ] Vídeo ≤ 15 min (roteiro em `VIDEO_SCRIPT.md`)
# tech-5
