TECH CHALLENGE FASE 5


## TECH CHALLENGE

Tech Challenge é o projeto que engloba os conhecimentos obtidos em

todas as disciplinas dessa fase. Esta é uma atividade que, a princípio, deve ser desenvolvida em grupo. É importante atentar-se ao prazo de entrega, uma vez que essa atividade é obrigatória, valendo 90% da nota de todas as disciplinas da fase.

## DESAFIO

## Guardiã AI - Inteligência Artificial para Saúde e Segurança da Mulher

Uma instituição que atua no atendimento à mulher deseja melhorar seus

processos utilizando Inteligência Artificial.

Durante um atendimento, diferentes tipos de informações podem ser

recebidos, como dados cadastrais, informações clínicas, sintomas, respostas de formulários, relatos escritos, documentos e histórico de atendimentos.

Essas informações precisam ser analisadas por profissionais para

identificar situações que necessitam de maior atenção, orientar os próximos passos e consultar protocolos ou informações relacionadas ao atendimento.

O desafio será desenvolver a Guardiã AI, uma aplicação capaz de

receber informações de um atendimento e utilizar Inteligência Artificial para auxiliar na análise inicial do caso.

A solução deverá trabalhar com dois contextos: saúde da mulher e

segurança da mulher.

Na área de saúde, o grupo poderá trabalhar, por exemplo, com câncer de

mama, risco materno, complicações durante a gestação, saúde cardiovascular ou outro problema relacionado à saúde feminina.

Na área de segurança, a solução poderá analisar relatos, organizar

informações importantes, identificar situações que merecem atenção e consultar protocolos ou orientações previamente cadastradas.


Não é necessário resolver todos esses problemas. Cada grupo deverá

escolher um cenário principal e construir sua solução em torno dele.

Por exemplo, uma equipe poderá criar uma plataforma de triagem para

mulheres gestantes que, além de analisar indicadores de saúde, também permita interpretar um relato fornecido durante o atendimento.

Outra equipe poderá trabalhar com acompanhamento de mulheres em

tratamento de câncer de mama, utilizando Machine Learning para análise dos dados e uma LLM para explicar informações e consultar protocolos.

O mais importante é que o problema esteja relacionado ao tema proposto

e que a equipe consiga demonstrar uma jornada completa de utilização da solução.

## OBJETIVO

A aplicação deverá receber informações de uma pessoa atendida e

utilizar diferentes técnicas de Inteligência Artificial para produzir uma análise que possa apoiar um profissional.

Um possível fluxo seria:

Entrada dos dados → análise por Machine Learning → consulta de informações → interpretação utilizando LLM → apresentação dos

resultados para um profissional.

A decisão final sempre deverá permanecer com uma pessoa responsável

pelo atendimento.

A Guardiã AI não deverá realizar diagnóstico médico definitivo, prescrever

medicamentos ou tomar decisões relacionadas à segurança de forma automática.

## Desenvolvimento da solução

O projeto deverá utilizar um conjunto de dados relacionado ao problema

escolhido.


A equipe deverá explorar e preparar os dados, realizar o treinamento de

pelo menos dois modelos de Machine Learning e comparar seus resultados utilizando métricas adequadas ao problema, como accuracy, precision, recall ou F1-score.

Assim como trabalhado na primeira fase do curso, não basta apresentar

apenas a acurácia. O grupo deverá explicar por que determinadas métricas são mais importantes naquele cenário e discutir possíveis impactos de erros do modelo.

A solução também deverá apresentar alguma forma de interpretação do

resultado do modelo. Poderão ser utilizadas técnicas como feature importance, SHAP ou outra abordagem equivalente, permitindo compreender quais informações contribuíram para uma determinada predição.

Além do Machine Learning, a aplicação deverá possuir integração com

uma Large Language Model.

A LLM poderá ser utilizada para interpretar os resultados, resumir relatos,

responder perguntas ou transformar informações técnicas em uma explicação mais clara para o profissional responsável pelo atendimento.

A equipe também deverá utilizar alguma estratégia de RAG — Retrieval-

Augmented Generation.

Para isso, poderá ser criada uma pequena base de conhecimento

contendo protocolos, documentos públicos, cartilhas ou outros materiais relacionados ao problema escolhido.

Quando uma pergunta for realizada, a aplicação deverá consultar essa

base e utilizar os documentos encontrados como contexto para a resposta da LLM.

O ideal é que a aplicação também consiga indicar qual informação ou

documento foi utilizado para gerar a resposta.


Essa abordagem mantém a evolução iniciada nos desafios anteriores, nos

quais LLMs passam a ser utilizadas para interpretar resultados e, posteriormente, são integradas a fluxos de consulta e assistentes inteligentes.

A equipe deverá utilizar LangChain ou LangGraph para organizar o fluxo

principal da aplicação.

Não é necessário criar uma arquitetura complexa com diversos agentes.

Um fluxo simples já é suficiente, por exemplo:

Receber dados → executar modelo → consultar documentos → gerar

explicação → retornar resultado.

Caso o grupo queira trabalhar com agentes, isso poderá ser utilizado

como diferencial, mas não será obrigatório.

Da mesma forma, a utilização de áudio, imagem ou vídeo poderá ser

considerada como uma evolução adicional do projeto, seguindo os conceitos trabalhados na Fase 4, mas não precisa ser obrigatória para todos os grupos.

## Segurança e responsabilidade

Como o projeto envolve saúde e segurança da mulher, a equipe deverá

ter atenção especial à forma como a Inteligência Artificial será utilizada.

O projeto deverá utilizar preferencialmente dados públicos, anonimizados

ou sintéticos.

A aplicação deverá deixar claro que a IA funciona como um sistema de

apoio ao profissional, e não como substituição da decisão humana.

As respostas geradas pela LLM deverão evitar recomendações definitivas

quando não houver informação suficiente.

Também deverá existir algum tipo de registro das análises realizadas,

permitindo compreender quais informações foram utilizadas e qual resultado foi produzido.


Essa preocupação com limites de atuação, logging, auditoria e

explicabilidade já aparece como requisito importante na Fase 3 do curso.

## ENTREGÁVEIS

Ao final dos dois meses, cada grupo deverá entregar um repositório Git

contendo o código-fonte da solução, os notebooks ou scripts utilizados no treinamento dos modelos, instruções para execução e um README explicando o problema, a arquitetura utilizada e os principais resultados.

A aplicação deverá possuir pelo menos uma forma simples de execução,

podendo ser uma interface web, uma API ou uma aplicação utilizando ferramentas como Streamlit ou Gradio.

O projeto deverá possuir um Dockerfile, permitindo executar a solução

de forma containerizada.

Também deverá ser entregue um relatório técnico em PDF apresentando

o problema escolhido, os dados utilizados, o processo de preparação dos dados, os modelos testados, as métricas obtidas, a integração com a LLM, o funcionamento do RAG e as principais limitações da solução.

Por fim, o grupo deverá produzir um vídeo de até 15 (quinze) minutos,

publicado no YouTube ou Vimeo como público ou não listado, demonstrando a aplicação funcionando.

No vídeo deverá ser possível acompanhar uma jornada completa, desde

a entrada das informações até a análise realizada pela Inteligência Artificial e a apresentação do resultado.

## Resultado esperado

Ao final do Hackathon, espera-se que o grupo tenha desenvolvido um

pequeno produto de Inteligência Artificial que combine:

Dados + Machine Learning + LLM + RAG + aplicação.


O foco principal não será a quantidade de tecnologias utilizadas, mas a

capacidade da equipe de construir uma solução coerente, funcional e responsável para um problema relacionado à saúde e segurança da mulher.

A solução deverá demonstrar que os alunos conseguem sair de um

modelo isolado de Inteligência Artificial e transformá-lo em uma aplicação capaz de apoiar uma situação real.


Tech Challenge

Página 8 de 8
