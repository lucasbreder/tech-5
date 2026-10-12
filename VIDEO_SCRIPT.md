# Roteiro do vídeo — Guardiã AI (até 15 minutos)

Roteiro de gravação com **inserts animados** (Remotion) intercalados à fala. Os inserts ficam em
`video/out/` (`I1_*.mp4` … `I9_*.mp4`), não têm áudio e usam 1920×1080 a 30 fps — grave a sua
fala separada e sobreponha, ou fale em cima do insert.

Este arquivo tem **duas partes**:

1. **Montagem** — onde encaixar cada insert, o que mostrar na tela e a linha do tempo.
2. **Fala do locutor (teleprompter)** — o texto corrido para ler, já alinhado aos inserts.

> Como funciona a montagem: você grava a **fala em câmara** e a **tela do app rodando**
> (`python main.py demo` → http://localhost:8501). Nos pontos marcados com **▶ INSERT**, faça um
> corte para o vídeo do insert. Os inserts explicam o que a fala acabou de dizer — deixe a narração
> continuar por cima ou pause ~2 s no fim do insert antes de voltar para a câmara/tela.

> Ritmo sugerido para o locutor: **~150 palavras por minuto**. Se sobrar tempo, alongue o bloco da
> jornada (tela do app); se faltar, corte exemplos, não a explicação dos modelos.

## Índice dos inserts

| Arquivo | Insert | Duração | Assunto |
| --- | --- | --- | --- |
| `I1_abertura.mp4` | I1 Abertura | 8 s | Marca, cenário e o limite "não diagnostica/não decide" |
| `I2_arquitetura.mp4` | I2 Arquitetura | 13 s | Fluxo ponta a ponta (dados → ML → RAG → LLM → auditoria) |
| `I3_dados.mp4` | I3 Dados | 11 s | Funil 1.014 → 451 → 380 e unidades clínicas |
| `I4_modelos.mp4` | I4 Modelos | 14 s | 3 classificadores, recall da classe alto, falso negativo |
| `I5_rag.mp4` | I5 RAG + LLM | 12 s | Protocolos públicos, vector store, llama3 local |
| `I6_relato.mp4` | I6 Relato | 10 s | Termos do relato organizam a escuta, não confirmam violência |
| `I7_auditoria.mp4` | I7 Auditoria | 9 s | JSONL de rastreabilidade de cada análise |
| `I8_docker.mp4` | I8 Docker/testes | 8 s | Container, `pytest` e relatório PDF |
| `I9_limites.mp4` | I9 Limitações | 10 s | O que a ferramenta não pode afirmar + encerramento |

## Linha do tempo sugerida

| Tempo | Cena | Fonte de imagem |
| --- | --- | --- |
| 0:00 | Abertura | **▶ INSERT I1** sobre a fala de abertura |
| 0:08 | Problema e cenário | Câmara |
| 0:40 | Visão do fluxo | **▶ INSERT I2** (fale por cima) |
| 1:00 | Dados e preparação | Tela (notebook / `DATA_SOURCE.md`) |
| 1:15 | Tamanho e limpeza do conjunto | **▶ INSERT I3** |
| 1:45 | Preparação e splits | Tela (notebook) |
| 3:00 | Rodar `python main.py train` | Tela (terminal) |
| 3:20 | Métricas e justificativa | **▶ INSERT I4** |
| 4:00 | Matriz de confusão, importância, SHAP | Tela (terminal/arquivos) |
| 5:00 | Base de conhecimento e Ollama | Tela (`data/knowledge/`, `ollama list`) |
| 5:10 | RAG + LLM local | **▶ INSERT I5** |
| 6:30 | Jornada no Streamlit | Tela (`localhost:8501`) |
| 9:00 | Caso “Alterações e relato sensível” | **▶ INSERT I6** e volte à tela |
| 11:30 | Relatório e log de auditoria | Tela (download + JSONL) |
| 11:40 | Rastreabilidade | **▶ INSERT I7** |
| 13:00 | Docker e testes | Tela (`docker compose up --build`, `pytest -q`) |
| 13:10 | Ambiente reprodutível | **▶ INSERT I8** |
| 14:00 | Limitações e encerramento | **▶ INSERT I9** |
| 14:20 | Fechamento | Câmara |

---

# Parte 1 — Montagem (onde cortar)

### 0:00–1:00 — Problema
- **▶ INSERT I1 (8 s)** na abertura; comece a falar em cima.
- **▶ INSERT I2 (13 s)** ao apresentar a visão geral do fluxo; narre por cima cada bloco.

### 1:00–3:00 — Dados e preparação
- Mostrar `data/raw/DATA_SOURCE.md` e o notebook.
- **▶ INSERT I3 (11 s)** ao explicar o tamanho do conjunto (1.014 → 451 → 380).
- Citar as limitações do UCI (sem idade gestacional, exames ou história obstétrica).

### 3:00–5:00 — Modelos e métricas
- Executar/mostrar `python main.py train`.
- **▶ INSERT I4 (14 s)** ao comparar os três classificadores e justificar o recall.
- Voltar à tela: matriz de confusão, importância e SHAP.

### 5:00–6:30 — RAG e LLM local
- Mostrar as duas notas em `data/knowledge/`, seus links e `ollama list`; rodar `rag-ingest`.
- **▶ INSERT I5 (12 s)** ao explicar base + vector store + llama3 local.

### 6:30–11:30 — Jornada completa
- Abrir `python main.py demo` (http://localhost:8501) com os três presets.
- **▶ INSERT I6 (10 s)** no caso “Alterações e relato sensível”.

### 11:30–13:00 — Rastreabilidade
- Baixar relatório Markdown, abrir “Registro das análises”; **▶ INSERT I7 (9 s)**.

### 13:00–14:00 — Docker e testes
- `docker compose up --build`, `pytest -q`; **▶ INSERT I8 (8 s)**.

### 14:00–15:00 — Limitações
- **▶ INSERT I9 (10 s)** e encerramento em câmara.

---

# Parte 2 — Fala do locutor (teleprompter)

> Leia o texto abaixo. As marcas **[INSERT …]** indicam o corte; a fala continua por cima do insert.
> Cada linha é uma respiração — não é necessário ler tudo de uma vez.

## 0:00 — Abertura e problema

**[INSERT I1 — 8 s]**

> Olá, eu sou o(a) [seu nome] e este é o projeto Guardiã AI,
> desenvolvido para o Tech Challenge da Fase 5.
> A proposta nasce de uma pergunta simples:
> e se a triagem de risco materno pudesse chegar ao profissional já organizada,
> com os sinais do atendimento interpretados e com o protocolo certo ao lado?

> A Guardiã AI recebe os dados de um atendimento de gestante
> e, opcionalmente, um relato escrito,
> e devolve uma leitura de apoio.
> Mas é importante dizer desde já:
> a ferramenta não dá diagnóstico, não prescreve
> e não toma nenhuma decisão de segurança.
> Ela apoia o profissional — a decisão final é sempre humana.

## 0:40 — Visão do fluxo

**[INSERT I2 — 13 s]**

> Aqui está a visão geral do fluxo.
> Tudo começa no atendimento, com os sinais vitais e o relato opcional.
> A partir daí, o Machine Learning entra com três classificadores
> e um detector de combinação incomum.
> Em paralelo, o RAG consulta protocolos públicos
> e devolve os trechos com a fonte.
> Uma LLM local, rodando na própria máquina,
> escreve a nota de apoio citando esses protocolos.
> O resultado aparece no Streamlit,
> e cada análise fica registrada num log de auditoria.
> Tudo isso é orquestrado pelo LangGraph, num fluxo único.

## 1:00 — Dados e preparação

**[mostrar notebook e DATA_SOURCE.md]**

> Vamos aos dados.
> Usamos o conjunto público Maternal Health Risk, da UCI —
> dados anonimizados, então nenhum dado real de paciente.
> O arquivo bruto tem mil e catorze linhas e seis variáveis clínicas:
> idade, pressão sistólica, pressão diastólica,
> glicemia, temperatura e frequência cardíaca.

**[INSERT I3 — 11 s]**

> A preparação foi feita com cuidado.
> Primeiro, convertemos as unidades para a linguagem do atendimento:
> temperatura de Fahrenheit para Celsius,
> e glicemia de milimol por litro para miligrama por decilitro.
> Também traduzimos os rótulos para baixo, moderado e alto.

> Depois, cuidamos da qualidade.
> Removemos um registro com frequência cardíaca de sete batimentos por minuto,
> impossível clinicamente.
> Removemos todas as duplicatas —
> e isso é essencial, porque duplicatas antes da separação de treino e teste
> causariam vazamento de dados.
> E havia trinta e cinco grupos de sinais idênticos com rótulos de risco diferentes.
> Sem história clínica para desempatar, escolher um rótulo seria arbitrário,
> então removemos esses setenta e um registros.
> No fim, o conjunto cai de mil e catorze para quatrocentos e cinquenta e uma linhas,
> e depois para trezentas e oitenta.
> É um conjunto menor, e assumimos isso como limitação.

## 3:00 — Modelos e métricas

**[rodar/mostrar `python main.py train`]**

> Agora, os modelos.
> Comparamos três classificadores:
> regressão logística, random forest e gradient boosting.

**[INSERT I4 — 14 s]**

> E aqui está a decisão mais importante do projeto: qual métrica priorizar?
> Em risco materno, o erro que mais preocupa é o falso negativo —
> quando o modelo diz “risco baixo”
> e deixa de destacar um caso que precisava de atenção.
> Por isso, o critério principal é o recall da classe alto:
> a capacidade de encontrar os casos graves.
> Em caso de empate, decidimos pela precisão da classe alto.

> A escolha é feita na validação cruzada de cinco folds, só no treino,
> e o resultado final é medido uma única vez no conjunto de teste.
> A random forest venceu:
> recall alto médio de zero vírgula setecentos e noventa e três na validação,
> e noventa e cinco por cento no teste.
> A gradient boosting ficou com noventa por cento,
> e a regressão logística com setenta e cinco.
> Também usamos um Isolation Forest, à parte,
> só para marcar quando a combinação de sinais é incomum em relação ao treino.
> E, para interpretabilidade,
> mostramos a importância global das variáveis
> e o efeito local de cada valor neste atendimento específico.

## 5:00 — RAG e LLM local

**[mostrar `data/knowledge/`, links oficiais e `ollama list`]**

> Chegamos à camada de conhecimento.
> A Guardiã AI não inventa orientação:
> ela consulta uma base construída a partir de dois documentos públicos
> do Ministério da Saúde —
> o Manual de Gestação de Alto Risco, de 2022,
> e o Guia de atendimento a mulheres em situação de violência doméstica, de 2025.
> Cada nota guarda a URL oficial.

**[INSERT I5 — 12 s]**

> Esses documentos passam por chunking e viram embeddings,
> guardados num vector store Chroma.
> Na hora da análise, o retriever devolve o trecho mais relevante
> junto com a fonte e a pontuação de relevância.
> A nota de apoio é escrita por uma LLM local,
> o llama3 via Ollama —
> sem chave de API e sem mandar dado para fora da máquina.
> O prompt é construído para o modelo citar o protocolo usado
> e nunca prescrever.
> E o relato e os documentos são tratados como dados não confiáveis,
> como proteção contra injeção de prompt.

## 6:30 — Jornada completa

**[abrir `python main.py demo` — http://localhost:8501]**

> Vamos ver a jornada completa na prática.
> Aqui está a interface.
> À esquerda, escolhemos o modelo e vemos a comparação de métricas.
> No centro, informamos os sinais do atendimento, com as unidades explícitas,
> e o relato é opcional.

> Vou começar com o exemplo “Sinais estáveis”.
> **[analisar]**
> Vejam: risco baixo, com a probabilidade do modelo,
> a distribuição entre as classes
> e os três classificadores lado a lado.
> Abaixo, o peso global das variáveis e o efeito local.

> Agora troco para “Pressão e glicemia em atenção”.
> **[analisar]**
> O modelo passa para risco alto.
> Reparem na pressão sistólica de cento e quarenta e oito
> e na glicemia de cento e sessenta e oito —
> os valores que mais empurram a probabilidade para cima.
> Aqui também aparecem a faixa de atenção e o Isolation Forest.
> E, para cada afirmação, os protocolos consultados,
> com o trecho e o link oficial.

> Agora o terceiro exemplo: “Alterações e relato sensível”.

**[INSERT I6 — 10 s]**

> Aqui entra a camada de segurança da mulher.
> O relato menciona medo, ameaça, agressão
> e um pedido para não ligar para o parceiro.
> A aplicação destaca esses termos —
> mas é fundamental entender o que isso significa.
> A presença de um termo não confirma que houve violência;
> e a ausência de um termo também não confirma segurança.
> A ferramenta apenas organiza a escuta.
> Ela não liga, não denuncia e não aciona ninguém.
> Quem avalia a situação é o profissional.

## 11:30 — Rastreabilidade

**[baixar o relatório Markdown e abrir “Registro das análises”]**

> Toda análise é registrada.
> O profissional pode baixar o relatório do atendimento em Markdown,
> e cada execução vai para um log de auditoria em JSONL.

**[INSERT I7 — 9 s]**

> O registro guarda o horário, os dados informados,
> a classe do modelo, os documentos consultados e o texto da nota.
> Isso dá rastreabilidade e explicabilidade.
> Um ponto importante: esse log não é prontuário
> e nunca recebe identificadores reais —
> trabalhamos só com dados de demonstração.

## 13:00 — Docker e testes

**[mostrar `docker compose up --build` e `pytest -q`]**

> O projeto é executável de ponta a ponta.
> Com Docker Compose, um comando sobe a aplicação
> já com os modelos treinados e os protocolos indexados.

**[INSERT I8 — 8 s]**

> Também temos testes automatizados com pytest,
> e um relatório técnico completo em PDF,
> com o problema, os dados, os modelos, as métricas,
> o RAG, a LLM e as limitações.

## 14:00 — Limitações e encerramento

**[INSERT I9 — 10 s]**

> Para fechar, quero ser honesto sobre os limites.
> O recall da classe alto não é cem por cento,
> então falsos negativos são possíveis.
> A classe moderado é a mais fraca e ainda precisa de atenção.
> As probabilidades do modelo não são calibradas como risco clínico individual.
> O dataset é pequeno e não tem idade gestacional nem exames.
> A busca é lexical, e a LLM local pode errar.

**[câmara]**

> Por tudo isso, a Guardiã AI é uma ferramenta de apoio:
> os protocolos e o julgamento humano sempre prevalecem.
> A decisão final é sempre humana.
> Obrigado.

---

# Dicas de gravação

- Grave a tela do app em 1080p (OBS). Use os presets da barra lateral para variar os casos.
- Se falar por cima dos inserts, comece a frase ~0,5 s antes do corte para evitar silêncio.
- Como os inserts não têm áudio, o alinhamento é livre — ajuste para caber no tempo de 15 min.
- Números por extenso (ex.: “cento e quarenta e oito”) ajudam o locutor a não travar.

# Como visualizar / renderizar os inserts

```bash
cd video
npm install
npm run studio            # pré-visualizar todas as cenas no Remotion Studio
npm run render:inserts    # gera out/I1_...mp4 até out/I9_...mp4
npm run render:all        # gera out/guardia_ai_completo.mp4 (todos em sequência)
npm run render:montagem   # junta inserts + gravações (fala normalizada + trilha) com crossfade
```

> Para a montagem final, coloque as gravações em `video/public/clips/` na ordem
> (`01-intro.mp4` … `09-final.mp4`). Os inserts entram como abertura de cada item, na ordem
> **Insert 1 → gravação 1 → … → Insert 9 → gravação 9**, com crossfade automático.
>
> **Áudio:** a fala é normalizada para ~-16 LUFS (`public/clips_norm/`) e há uma **trilha**
> (`public/music/background.mp3`) com *ducking*: **alta nos inserts** (motion graphics) e
> **baixa sob a fala**. Saída: `out/guardia_ai_final.mp4`.
