# Conjunto de dados

Arquivo: `Maternal Health Risk Data Set.csv`

Origem: UCI Machine Learning Repository, conjunto *Maternal Health Risk*.

https://archive.ics.uci.edu/dataset/863/maternal+health+risk

Colunas originais: `Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`, `RiskLevel`.

Nesta aplicação a preparação (`src/data/preprocessing.py`) converte temperatura de Fahrenheit para °C, glicemia de mmol/L para mg/dL, traduz o risco para `baixo` / `moderado` / `alto` e remove frequência cardíaca incompatível (há registros com 7 bpm). Duplicatas saem antes do split para não repetir o mesmo atendimento no treino e no teste.

Uso: dados públicos de demonstração. Não inserir prontuário real.
