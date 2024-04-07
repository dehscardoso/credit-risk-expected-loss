# Modelagem da Probabilidade de Risco de Crédito e Cálculo da Perda Esperada

No atual cenário econômico, a concessão de crédito emerge como uma decisão revestida de incerteza, configurando-se como um desafio substancial. Esta realidade aplica-se a uma variedade de contextos financeiros, incluindo empréstimos, financiamentos e transações comerciais a prazo. Inerente a essas operações está a potencialidade de ocorrência de perdas. Apesar da implementação de critérios rigorosos para a concessão de crédito, que abarcam a análise da saúde financeira do solicitante, a qualidade do relacionamento com o credor e o histórico de pagamentos prévios, persiste o risco de inadimplemento por parte do requerente. Tal inadimplemento acarreta a não recuperação do capital investido e dos juros correspondentes, culminando em interrupções nos fluxos de caixa e escalada nos custos associados à cobrança de dívidas (SICSÚ, 2010).

Contudo, a habilidade de estimar a probabilidade de risco de uma operação de crédito e a identificação prévia de solicitantes potencialmente inadimplentes podem conferir maior segurança à decisão de concessão de crédito, mitigando, assim, as perdas institucionais. Neste contexto, o objetivo principal deste projeto consistiu na modelagem de risco de crédito. Para tanto, foram empregadas metodologias de aprendizado de máquina e técnicas estatísticas avançadas, visando à identificação de variáveis preditoras de inadimplência e à estimativa da **Perda Esperada**.

Ressalta-se a importância de se aprofundar no entendimento do ciclo de crédito, o qual, apesar de apresentar variações conforme o modelo de negócio em questão, geralmente compreende as seguintes etapas.

<br>

## Dataset

Este projeto teve como objetivo medir o risco de crédito do conjunto de dados [Lending Club Loan Data](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1), que contém informações de empréstimos da empresa americana de empréstimo peer-to-peer Lending Club, através do cálculo da perda esperada de seus empréstimos pendentes.

<br>

## Desenvolvimento da Solução 

O desenvolvimento desta solução iniciou com a aplicação de uma metodologia criteriosa para a seleção de variáveis e a construção de um conjunto de dados robusto, empregando técnicas estatísticas avançadas. Esta fase inicial envolveu a seleção cuidadosa de um dataset representativo, seguida de uma análise detalhada das variáveis, utilizando métricas como o peso da evidência (*Weight of Evidence* - WoE) e o valor de informação (*Information Value* - IV). Após esta etapa de seleção, os dados foram organizados em categorias através das técnicas de coarse e fine classing, visando uma segmentação eficiente e significativa das variáveis.

A etapa subsequente consistiu em uma análise exploratória dos dados, com o objetivo de examinar a correlação e distribuição das variáveis selecionadas. Esta análise foi fundamental para obter insights valiosos sobre a estrutura dos dados, facilitando assim a modelagem subsequente. Com base nesses insights, procedeu-se ao desenvolvimento de um modelo de probabilidade de inadimplência empregando regressão logística, uma técnica que permite estimar a probabilidade de ocorrência de um evento, neste caso, a inadimplência, com base nas características dos mutuários.

O passo seguinte envolveu a elaboração de um sistema de credit scoring, derivado dos coeficientes obtidos no modelo de regressão logística. Esta etapa permitiu a classificação dos mutuários de acordo com o risco de inadimplência, possibilitando a aplicação prática do modelo em decisões de concessão de crédito.
Para assegurar a eficácia e a estabilidade do modelo ao longo do tempo, implementou-se um mecanismo de monitoramento baseado no Índice de Estabilidade Populacional (*Population Stability Index* - PSI). Além disso, realizou-se uma análise comparativa entre os dados utilizados no desenvolvimento do modelo e um novo conjunto de dados, visando validar a robustez e a generalidade do modelo proposto.

Paralelamente, modelos preditivos para o cálculo do índice de perda e da exposição ao risco em casos de inadimplência foram desenvolvidos utilizando regressão linear, complementando a análise de risco de crédito. Finalmente, a perda esperada foi calculada integrando os resultados dos três modelos: probabilidade de inadimplência, índice de perda e exposição ao risco, oferecendo uma ferramenta abrangente e precisa para a gestão do risco de crédito.

Esta solução representa um avanço significativo na área de gestão de risco de crédito, combinando análise estatística rigorosa, modelagem preditiva e monitoramento contínuo para oferecer uma abordagem holística e eficaz na previsão e mitigação do risco de inadimplência.
