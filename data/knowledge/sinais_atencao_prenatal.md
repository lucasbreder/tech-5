# Nota de apoio — sinais para releitura no pré-natal

Documento elaborado para a base da Guardiã AI. Resume, em texto próprio, pontos de atenção usados na conversa com o profissional. Não substitui o pré-natal de risco do Ministério da Saúde nem qualquer protocolo institucional.

Fonte oficial: https://www.gov.br/saude/pt-br/composicao/saps/publicacoes/livros/manual-de-gestacao-de-alto-risco

Referência: Brasil. Ministério da Saúde. *Manual de Gestação de Alto Risco*. Brasília: Ministério da Saúde, 2022. Esta nota resume pontos do documento público; não o substitui.

## Pressão arterial

A pressão arterial é medida em milímetros de mercúrio, com o valor sistólico e o diastólico. Nesta aplicação, valores a partir de 140 mmHg de sistólica ou 90 mmHg de diastólica entram na faixa de atenção para o profissional reler o caso — junto com queixa de dor de cabeça, alteração visual ou inchaço. Isso não fecha diagnóstico de pré-eclâmpsia, hipertensão gestacional ou hipertensão crônica. Falta, entre outros dados, a idade gestacional, proteinúria, exames e a série de medidas.

O profissional confirma o manguito, o repouso e se a medida se repete. Uma leitura isolada na tela não autoriza medicação, internação ou alta.

## Glicemia

A glicemia desta aplicação está em mg/dL. O conjunto público original registra o dado em mmol/L; a preparação multiplica por 18 para a unidade usada no atendimento. Valor persistentemente elevado, sede, perda de peso ou antecedente de diabetes pedem revisão profissional e conferência do exame laboratorial. A tela não distingue jejum, pós-prandial ou teste de tolerância, portanto não classifica diabetes mellitus gestacional.

## Temperatura e frequência cardíaca

Temperatura em graus Celsius. A partir de cerca de 37,8 °C a aplicação marca o dado para releitura, porque febre na gestação pede investigação de foco infeccioso pelo profissional — não um palpite da ferramenta. Frequência cardíaca acima de 100 bpm ou abaixo de 60 bpm também é destacada. Taquicardia pode acompanhar febre, dor, ansiedade ou anemia; a causa não é decidida aqui.

## Como usar o resultado do modelo

O modelo devolve uma classe (baixo, moderado ou alto) aprendida em um conjunto público de risco materno com seis variáveis: idade, pressão sistólica, pressão diastólica, glicemia, temperatura e frequência cardíaca. A classe alto concentrar a revisão é desejável: um falso negativo atrasa o olhar humano. Um falso alarme gasta tempo da equipe e também precisa ser reconhecido. Nenhuma das duas saídas prescreve fármaco, dose, via de parto ou internação.

Perguntas úteis antes de concluir: a medida foi repetida? Há queixa nova? Qual a idade gestacional? O exame confirma a glicemia da tela? O modelo não viu esses itens.

## O que esta nota não faz

Não indica medicamento, não calcula escore obstétrico oficial e não substitui a classificação de risco da unidade. Serve para o profissional localizar, na mesma tela, o número que saiu da faixa e o trecho de protocolo correspondente.
