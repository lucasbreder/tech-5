# Guardiã AI — Relatório Técnico

Tech Challenge — Fase 5

> Sistema de apoio à decisão. A Guardiã AI não realiza diagnóstico, não prescreve medicamentos e não toma decisões automáticas de segurança. A avaliação final é sempre humana.

## 1. Problema e cenário

A solução apoia a triagem inicial de gestantes. O profissional informa idade, pressão arterial, glicemia, temperatura, frequência cardíaca e, opcionalmente, um relato escrito. A aplicação combina Machine Learning tabular, detecção de anomalias, RAG, uma LLM local e uma interface Streamlit.

O cenário principal é risco materno. A leitura transparente de termos do relato adiciona o contexto de segurança da mulher sem afirmar que houve violência.

## 2. Dados

O conjunto público é o UCI Maternal Health Risk Data Set:

https://archive.ics.uci.edu/dataset/863/maternal+health+risk

O arquivo bruto contém 1.014 linhas e sete colunas. Após remover duplicatas exatas e uma frequência cardíaca incompatível de 7 bpm, restam 451 linhas. Em seguida, saem 71 linhas de 35 grupos nos quais o mesmo vetor de sinais possuía rótulos diferentes. Sem história clínica para desempatar, escolher um rótulo seria arbitrário. Restam 380 linhas. A retirada ocorre antes da separação treino/teste.

Preparação:

- temperatura convertida de Fahrenheit para graus Celsius;
- glicemia convertida de mmol/L para mg/dL;
- rótulos traduzidos para baixo, moderado e alto;
- split estratificado de 80% para treino e 20% para teste;
- seleção com validação cruzada estratificada de cinco folds apenas no treino;
- padronização apenas dentro do pipeline da regressão logística.

Limitações dos dados: não há idade gestacional, histórico obstétrico, exames laboratoriais completos, repetição de pressão ou identificação da população de origem. Os registros não representam todas as gestantes brasileiras.

## 3. Modelos e métricas

Foram comparados três classificadores com a mesma partição de teste: regressão logística, random forest e gradient boosting.

- Random forest: accuracy 0,829; recall alto 0,950; precisão alto 0,950; F1 ponderado 0,832.
- Gradient boosting: accuracy 0,763; recall alto 0,900; precisão alto 0,947; F1 ponderado 0,743.
- Regressão logística: accuracy 0,711; recall alto 0,750; precisão alto 0,833; F1 ponderado 0,698.

O critério principal é o recall médio da classe alto nos cinco folds do treino. Um falso negativo pode deixar de destacar para revisão um caso de maior risco. Em empate, usa-se a precisão média da classe alto. A random forest foi escolhida por essa regra: recall alto médio 0,793 (desvio 0,156).

No teste final de 76 linhas, a random forest alcançou recall alto 0,950. O conjunto de teste é pequeno e não constitui validação clínica externa. O recall da classe moderado foi 0,667 e ainda exige atenção. Accuracy não é usada isoladamente.

## 4. Interpretabilidade

A aplicação mostra a importância global e uma explicação local: para cada variável, calcula a diferença na probabilidade da classe prevista ao trocar apenas aquele valor pela mediana do treino. O sinal indica se o valor informado aumenta ou reduz a saída em relação à referência.

O artefato SHAP contém a média do valor absoluto das contribuições na random forest. O TreeExplainer utilizado não suporta o gradient boosting multiclasse desta versão, por isso o relatório identifica explicitamente o modelo explicado.

Um Isolation Forest complementa a leitura e informa se a combinação dos sinais é incomum no conjunto de treino. Essa saída não é uma quarta classe clínica.

## 5. RAG

A base local contém notas rastreáveis que resumem dois documentos públicos:

- Manual de Gestação de Alto Risco, Ministério da Saúde, 2022;
- Guia prático para atendimento a mulheres em situação de violência doméstica na Atenção Primária à Saúde, Ministério da Saúde, 2025.

Cada nota guarda a URL oficial. O retriever local usa HashingVectorizer com unigramas e bigramas e armazena vetores no Chroma. A interface apresenta arquivo, relevância, trecho e link oficial. Uma consulta recupera o contexto clínico; outra pode recuperar o contexto do relato.

## 6. LLM e orquestração

LangGraph executa: entrada → classificadores → anomalia e relato → recuperação de documentos → explicação.

A LLM padrão é llama3 via Ollama, executada localmente, sem chave de API. O prompt trata relato e documentos como dados não confiáveis para reduzir prompt injection. A resposta recebe um aviso de segurança acrescentado pelo código, fora da geração. Se o Ollama falhar, uma nota determinística mantém a aplicação utilizável e registra o modo local.

## 7. Aplicação e auditoria

A interface Streamlit oferece exemplos, campos com unidades explícitas, comparação entre classificadores, distribuição da classe, importância, faixa de atenção, termos do relato, fontes RAG e download do relatório da análise.

Cada execução registra timestamp UTC, entrada, modelos, documentos consultados, nota e aviso em JSONL. O log serve para demonstração e auditoria técnica; não é prontuário. Não devem ser inseridos identificadores reais.

## 8. Segurança e responsabilidade

- não diagnostica nem prescreve;
- não aciona contato, denúncia ou medida protetiva;
- não conclui violência pela presença ou ausência de palavras;
- rejeita valores numéricos não finitos;
- mantém a decisão com o profissional;
- recomenda apenas dados públicos, fictícios ou anonimizados;
- explicita que a saída numérica do classificador não é probabilidade clínica calibrada.

## 9. Execução

Ambiente local:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve
ollama pull llama3
python main.py train
python main.py rag-ingest
python main.py demo
```

Docker Compose conecta o contêiner ao Ollama do host por host.docker.internal.

## 10. Limitações e evolução

- conjunto pequeno após remoção de duplicatas e conflitos;
- seleção por validação cruzada, mas avaliação final ainda sem validação externa;
- probabilidades não calibradas;
- busca lexical local, sem embedding semântico treinado em português;
- LLM local pode omitir, resumir incorretamente ou gerar texto inadequado;
- notas RAG não substituem a leitura integral dos documentos oficiais;
- faltam avaliação com profissionais, monitoramento em produção e governança clínica.

Evoluções possíveis: validação cruzada estratificada, calibração, dataset brasileiro validado, embeddings locais em português, testes de segurança da LLM e avaliação humana estruturada.
