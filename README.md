# Modelagem da Probabilidade de Risco de Crédito e Cálculo da Perda Esperada

Diante do cenário econômico atual, a concessão de crédito é uma decisão sob condições de incerteza e um grande desafio. Seja para empréstimos, financiamentos ou vendas a prazo, a possibilidade de perda sempre estará presente. Mesmo com critérios definidos para a concessão, como a análise da situação financeira, o relacionamento com o credor e o histórico de pagamentos, ainda há o risco do solicitante não cumprir com suas dívidas, fazendo com que um credor não receba o principal devido nem os juros, resultando na interrupção dos fluxos de caixa e no aumento dos custos de cobrança (SICSÚ, 2010).

No entanto, se o credor puder estimar a probabilidade de risco da operação e identificar esses requerentes, sua decisão poderá ser mais confiável, diminuindo perdas para a instituição. Portanto, o presente projeto teve como objetivo a modelagem de risco de crédito, utilizando algoritmos de machine learning e técnicas estatísticas, em que foram identificadas possíveis variáveis indicadoras de inadimplência e foi estimada a **Perda Esperada**.

<br>

## Dataset

Este projeto teve como objetivo medir o risco de crédito do conjunto de dados [Lending Club Loan Data](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1), que contém informações de empréstimos da empresa americana de empréstimo peer-to-peer Lending Club, através do cálculo da perda esperada de seus empréstimos pendentes.

<br>

## Desenvolvimento da Solução (corrigir)

Inicialmente, foi aplicada uma metodologia para seleção de features e criação de um dataset, utilizando técnicas estatísticas, em que foi selecionado um conjunto de dados e uma análise das suas variáveis foi realizada com base no peso da evidência e valor de informação, após a seleção os dados foram agrupados em classes com coarse e fine classing.

Em seguida, realizou-se uma análise exploratória dos dados para verificar a correlação e distribuição das features, o que possibilitou uma melhor compreensão dos dados. Com isso, foi desenvolvido um modelo de probabilidade de inadimplência com regressão logística.

Após isso, um score de crédito foi desenvolvido com base nos coeficientes das características e os mutuários foram pontuados. Também foi desenvolvido um monitoramento para o modelo de probabilidade de inadimplência, com o índice de estabilidade populacional e, realizou-se uma análise demonstrativa comparando os dados de desenvolvimento do modelo e de um novo conjunto de dados.

Os modelos de índice de perda e exposição por inadimplência com regressão linear foram criados e, por fim, a perda esperada foi calculada utilizando os três modelos.

De acordo com os resultados, a perda esperada do conjunto de dados selecionado possui uma média de US\$ 592,28, desvio padrão de US\$ 1.639,46 e valor máximo de US\$ 37.975,99. A soma total da perda esperada do portfólio foi de US\$ 519.819.141,88 e a proporção da perda esperada em relação ao valor dos empréstimos foi de 3,8%.
