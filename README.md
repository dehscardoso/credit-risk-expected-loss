<p align="center">
  <img src = "https://thumbs2.imgbox.com/32/93/xRH3CBEG_t.png" width="250">
</p>

<br> 

# Modelagem da Probabilidade de Risco de Crédito e Cálculo da Perda Esperada

Diante do cenário econômico atual, a concessão de crédito é uma decisão sob condições de incerteza e um grande desafio. Sejam empréstimos, financiamentos ou vendas a prazo, a possibilidade de perda sempre estará presente. Mesmo com critérios para a concessão definidos, como a análise da situação financeira, relacionamento com o credor e o histórico de pagamentos, ainda há risco do solicitante não cumprir com suas dívidas, fazendo com que um credor não receba o principal devido nem os juros, o que, por conseguinte, resulta na interrupção dos fluxos de caixa e no aumento dos custos de cobrança. 

No entanto, caso ele consiga estimar a probabilidade de risco da operação e identificar esses requerentes, sua decisão poderá ser mais confiável reduzindo perdas para a instituição. Portanto, o presente projeto teve como objetivo a modelagem de risco de crédito, utilizando algoritmos de machine learning, em que foram identificadas possíveis variáveis indicadoras de inadimplência e se foi estimada a **Perda Esperada**.

<br>

## Dataset

Este projeto mediu o risco de crédito do dataset [Lending Club Loan Data](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1), uma empresa americana de empréstimo peer-to-peer, calculando a perda esperada de seus empréstimos pendentes. 

<br>

## Desenho da Solução

Primeiramente foram aplicadas duas metodologias para seleção de features e criação de dois datasets: A primeira abordagem mais comumente utilizada por estatísticos, em que um conjunto de dados foi escolhido e realizada uma análise das suas variáveis com base no peso de evidência e valor de informação, após essa seleção os dados foram agrupados em classes com coarse e fine classing. Na outra, o conjunto dados foi selecionado com o algoritmo LightGBM, em que as features foram escolhidas por seu ganho no modelo.

Verificou-se por meio de uma análise exploratória dos dados nos dois datasets a correlação e distribuição das features que possibilitou uma melhor compreensão dos dados. 

Com isso, desenvolveu-se o modelo de probabilidade de inadimplência com regressão logística e realizada uma comparação de performance entre os dois datasets. O primeiro modelo foi selecionado para o desenvolvimento das demais etapas, apesar do melhor desempenho do segundo, que utiliza LightGBM.

Em seguida, um score de crédito foi desenvolvido com base nos coeficientes das features e os mutuários foram escorados. Também foi desenvolvido um monitoramento para o modelo de probabilidade de inadimplência, com o índice de estabilidade populacional e realizou-se uma análise demonstrativa comparando os dados de desenvolvimento do modelo e de um novo dataset. 

Os modelos de índice de perda e exposição por inadimplência com regressão linear foram criados e, por fim, a perda esperada foi calculada utilizando os três modelos.
