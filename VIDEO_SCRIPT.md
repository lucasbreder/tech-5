# Roteiro do vídeo — Guardiã AI (até 15 minutos)

## 0:00–1:00 — Problema

- Apresentar a triagem de risco materno com leitura opcional de relato.
- Reforçar: apoio ao profissional, sem diagnóstico, prescrição ou decisão automática.

## 1:00–3:00 — Dados e preparação

- Mostrar `data/raw/DATA_SOURCE.md` e o notebook.
- Explicar unidades, remoção de duplicatas e split estratificado.
- Citar as limitações do conjunto UCI.

## 3:00–5:00 — Modelos e métricas

- Executar ou mostrar `python main.py train`.
- Comparar os três classificadores.
- Justificar recall da classe alto e o desempate por precisão alto.
- Mostrar matriz de confusão, importância e SHAP.

## 5:00–6:30 — RAG e LLM local

- Mostrar as duas notas em `data/knowledge/` e seus links oficiais.
- Executar `python main.py rag-ingest`.
- Mostrar `ollama list`; explicar que llama3 roda localmente sem chave.
- Explicar o fluxo LangGraph.

## 6:30–11:30 — Jornada completa

- Abrir `python main.py demo`.
- Usar “Sinais estáveis” e analisar.
- Usar “Pressão e glicemia em atenção”.
- Mostrar classe, distribuição, comparação, importância, anomalia e fontes.
- Usar “Alterações e relato sensível”.
- Explicar que termos do relato organizam a escuta e não confirmam violência.
- Abrir um trecho RAG e o documento oficial.

## 11:30–13:00 — Rastreabilidade

- Baixar o relatório Markdown do atendimento.
- Abrir “Registro das análises”.
- Mostrar o JSONL sem utilizar dados reais de paciente.

## 13:00–14:00 — Docker e testes

- Mostrar `docker compose up --build`.
- Executar `pytest -q`.
- Citar o relatório técnico PDF.

## 14:00–15:00 — Limitações

- Recall alto abaixo de 1, classe moderado fraca e probabilidades não calibradas.
- Dataset pequeno, sem idade gestacional e exames.
- LLM pode errar; protocolos e avaliação humana prevalecem.
