# Modelagem da Probabilidade de Risco de Crédito e Cálculo da Perda Esperada

Diante do cenário econômico atual, a concessão de crédito é uma decisão sob condições de incerteza e um grande desafio. Sejam empréstimos, financiamentos ou vendas a prazo, a possibilidade de perda sempre estará presente. Mesmo com critérios para a concessão definidos, como a análise da situação financeira, relacionamento com o credor e o histórico de pagamentos, ainda há risco do solicitante não cumprir com suas dívidas, fazendo com que um credor não receba o principal devido nem os juros, o que, por conseguinte, resulta na interrupção dos fluxos de caixa e no aumento dos custos de cobrança. 

No entanto, caso ele consiga estimar a probabilidade de risco da operação e identificar esses requerentes, sua decisão poderá ser mais confiável reduzindo perdas para a instituição. Portanto, o presente projeto teve como objetivo a modelagem de risco de crédito, utilizando algoritmos de machine learning e técnicas estatísticas, em que foram identificadas possíveis variáveis indicadoras de inadimplência e se foi estimada a **Perda Esperada**.

<br>

## Dataset

Este projeto mediu o risco de crédito do dataset [Lending Club Loan Data](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1), uma empresa americana de empréstimo peer-to-peer, calculando a perda esperada de seus empréstimos pendentes. 

<br>

## Desenho da Solução

Primeiramente foi aplicada uma metodologia para seleção de features e criação de um dataset que utiliza técnicas estatísticas, em que um conjunto de dados foi escolhido e realizada uma análise das suas variáveis com base no peso de evidência e valor de informação, após essa seleção os dados foram agrupados em classes com coarse e fine classing. 

Verificou-se, por meio de uma análise exploratória dos dados no dataset, a correlação e distribuição das features, que possibilitou uma melhor compreensão dos dados. 

Com isso, desenvolveu-se o modelo de probabilidade de inadimplência com regressão logística e o primeiro modelo foi desenvolvimento das demais etapas.

Em seguida, um score de crédito foi desenvolvido com base nos coeficientes das features e os mutuários foram escorados. Também foi desenvolvido um monitoramento para o modelo de probabilidade de inadimplência, com o índice de estabilidade populacional e, realizou-se uma análise demonstrativa comparando os dados de desenvolvimento do modelo e de um novo dataset. 

Os modelos de índice de perda e exposição por inadimplência com regressão linear foram criados e, por fim, a perda esperada foi calculada utilizando os três modelos.

E, de acordo com os resultados, a perda esperada do dataset selecionado possui média de U\$ 592.28, desvio padrão de U\$ 1.639.46, e valor máximo de U\$ 37.975,99. A soma total da perda esperada do portifólio foi de U\$ 519.819.141,88 e, por fim, proporcão da perda esperada pelo valor dos empréstimo foi 3.8%.
