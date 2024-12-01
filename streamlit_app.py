import streamlit as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Configurar título e ícone da página
st.set_page_config(page_title="Previsão do Preço do Petróleo Brent", page_icon="🛢️")

# Carregar o modelo LSTM salvo
model = load_model('lstm_model.keras')

# Carregar os dados do CSV para ajustar o scaler
df_close = pd.read_csv('dados_petroleo.csv')

# Converter a coluna de data para datetime
df_close['Date'] = pd.to_datetime(df_close['Date'])

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_close[['Close']])

# Defina o número de dias históricos a serem usados para a previsão
NUM_DAYS = 60  # Ajuste conforme necessário

# Limite o horizonte de previsão
MAX_FORECAST_DAYS = 15  # Limite de 15 dias para previsão

# Calcular a data máxima permitida para previsão
last_date = df_close['Date'].max()
max_forecast_date = last_date + timedelta(days=MAX_FORECAST_DAYS)

# Função para verificar se uma data é um dia útil
def is_weekday(date):
    return date.weekday() < 5  # 0 é segunda-feira, 4 é sexta-feira

# Função para gerar previsões a partir de um número configurável de dias até a data selecionada
def generate_predictions(df, model, scaler, end_date, num_days):
    predictions = []
    
    # Prepare uma sequência de dados para prever
    last_days = df[['Close']].values[-num_days:]  # Pega os últimos 'num_days' dias
    last_days_scaled = scaler.transform(last_days)
    
    # Converter end_date para Timestamp
    end_date = pd.Timestamp(end_date)
    
    current_date = df['Date'].max() + timedelta(days=1)
    
    while current_date <= end_date:
        if is_weekday(current_date):
            # Adicionar a previsão ao final da sequência
            date_array_scaled = np.array([last_days_scaled])
            
            # Fazer a previsão
            prediction_scaled = model.predict(date_array_scaled)
            
            # Reverter a normalização
            prediction = scaler.inverse_transform(
                np.concatenate((prediction_scaled, np.zeros((prediction_scaled.shape[0], df.shape[1] - 1))), axis=1)
            )[:, 0]
            
            predictions.append((current_date, prediction[0]))
            
            # Atualizar a sequência de dados para a próxima previsão
            last_days_scaled = np.append(last_days_scaled[1:], prediction_scaled, axis=0)
        
        current_date += timedelta(days=1)
    
    return predictions

# Configuração do menu lateral
st.sidebar.title("Navegação")
menu = st.sidebar.radio("Ir para", ["Home", "Análise Exploratória dos Dados", "Modelo Preditivo", "Dashboard - Exploração e Insights", "MVP e Plano de Deploy", "Previsão do Preço do Petróleo", "Conclusão"])

# Seções da aplicação
if menu == "Home":
    st.title("Previsão do Preço do Petróleo Brent")
 
    st.image("imagens/pump-jack-848300_640.jpg")

    st.write(""" 
    O ambiente de negócios global é caracterizado por sua complexidade e dinamismo, onde commodities, como o petróleo Brent, desempenham um papel bastante relevante. As flutuações nos preços dessas commodities são influenciadas por uma variedade de fatores, incluindo decisões políticas e econômicas, eventos geopolíticos e mudanças na demanda global de energia. A volatilidade dos preços do petróleo é uma característica intrínseca desse mercado, intensificada por crises internacionais, conflitos regionais e alterações nas políticas energéticas. Tais eventos podem resultar em variações significativas nos preços, tornando a previsão de seu comportamento um desafio complexo.

    Neste contexto, desenvolvemos um modelo preditivo para antecipar as variações diárias no preço do petróleo Brent. Para atingir esse objetivo, utilizamos técnicas de Machine Learning e Análise de Séries Temporais, baseando-nos em dados históricos disponíveis. Através dessa abordagem, procuramos identificar tendências e padrões no mercado de petróleo, visando antecipar seus movimentos futuros e fornecer insights valiosos para a tomada de decisão.

    Para a modelagem preditiva, implementamos o LSTM (Long Short-Term Memory). A avaliação do modelo foi realizada utilizando métricas de desempenho como MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error) e R² (Coeficiente de Determinação). Através dessas métricas, buscamos atestar a precisão e confiabilidade para a previsão dos preços do petróleo.
             
    Além disso, incluimos a criação de um dashboard que oferece uma visão abrangente sobre a variação dos preços do petróleo, destacando insights relevantes relacionados a fatores geopolíticos, crises econômicas e demanda energética global. Esse dashboard foi desenvolvido utilizando Power BI, permitindo uma análise visual intuitiva e detalhada.

    """)

elif menu == "Análise Exploratória dos Dados":
    st.title("Análise Exploratória dos Dados")
    
    st.subheader("Coleta e Avaliação dos Dados")
    st.write("""
    Para a construção de um modelo preditivo robusto e confiável, a etapa inicial de coleta de dados é fundamental. Neste projeto, visando obter um conjunto de dados históricos do Preço por Barril do Petróleo Brent, exploramos duas principais fontes de dados: o site ipeadata.gov.br e a biblioteca yfinance, disponibilizada pelo portal Yahoo! Finance. Ambas as fontes oferecem acesso a dados históricos abrangentes do mercado financeiro, incluindo o Preço por Barril do Petróleo Brent.
    
    **Avaliação das Fontes de Dados**
    
    O site ipeadata.gov.br é uma plataforma mantida pelo Instituto de Pesquisa Econômica Aplicada (Ipea), uma instituição de destaque no Brasil. A plataforma oferece uma ampla gama de dados econômicos, sociais e financeiros, que são essenciais para análises e pesquisas, bem como para a formulação de políticas públicas. Entre os diversos indicadores disponíveis, o site fornece dados sobre o Preço por Barril do Petróleo Brent, com uma granularidade diária, permitindo um acompanhamento detalhado das flutuações de mercado e facilitando estudos econômicos aprofundados.
    
    Por outro lado, a biblioteca yfinance emergiu como uma alternativa eficaz para a obtenção de dados financeiros, permitindo a integração direta com modelos de análise em Python. Essa biblioteca facilita a extração de informações do portal Yahoo! Finance, incluindo dados detalhados sobre o Preço por Barril do Petróleo Brent. Uma das principais vantagens da utilização do yfinance é a simplificação do processo de coleta e tratamento dos dados, já que eles são apresentados em um formato conveniente para análises subsequentes.
    
    **Verificação da Consistência dos Dados**
    
    Realizou-se testes comparativos entre os dados fornecidos pelo ipeadata.gov.br e aqueles obtidos através da yfinance. Essa análise revelou uma alta concordância entre os conjuntos de dados, confirmando a confiabilidade e a precisão das informações obtidas por meio da biblioteca yfinance.
    
    **Decisão e Implementação**
    
    Considerando a equivalência dos dados e a eficiência operacional, optou-se por utilizar a biblioteca yfinance como fonte primária de dados históricos do Preço por Barril do Petróleo Brent. Essa escolha foi motivada, principalmente, pela facilidade de integração e pelo menor esforço necessário no tratamento dos dados.
    """)

    st.subheader("Conjunto de Dados")
    st.write("""
    Definiu-se o escopo da coleta de dados para um período de 01/01/2019 a 29/11/2024, com frequência diária. Os dados históricos coletados a partir da biblioteca yfinance incluem as seguintes colunas:
    
    - **Date (Data):** Indica a data de cada sessão de negociação à qual os dados daquela linha (Open, High, Low, Close, Adj Close, Volume) se referem.
    - **Open (Abertura):** Reflete o preço do barril de petróleo Brent no início do dia de mercado. É um indicador importante para sentir o clima inicial do mercado de petróleo.
    - **High (Alta):** Este é o preço máximo que o barril de petróleo Brent alcançou no decorrer do dia. É útil para entender até que ponto os investidores estavam dispostos a comprar.
    - **Low (Baixa):** Indica o preço mais baixo que o barril de petróleo Brent atingiu durante o dia. Ajuda a avaliar o nível de venda ou pessimismo no mercado.
    - **Close (Fechamento):** Mostra o preço em que o barril de petróleo Brent se estabilizou ao final do dia de negociações. É um dos indicadores mais observados, pois reflete o consenso do mercado ao final do dia.
    - **Adj Close (Fechamento Ajustado):** Para commodities como o petróleo, este valor geralmente espelha o fechamento, servindo como uma referência final do estado do mercado, ajustando-se para fatores como ajustes de contratos futuros.
    - **Volume:** Representa o total de contratos de petróleo Brent negociados durante o dia. Um volume alto pode indicar um grande interesse ou uma mudança significativa no mercado, enquanto um volume baixo pode sugerir o contrário.
    
    Cada uma dessas colunas fornece percepções valiosas sobre o comportamento do mercado de petróleo Brent durante o período especificado, permitindo análises detalhadas sobre tendências, volatilidade e interesse dos investidores. Isso facilita o esforço necessário no tratamento dos dados para gerar insights significativos sobre o mercado de petróleo.
    """)

    st.subheader("Análise Exploratória dos Dados")
    st.write("""
    A organização prévia dos dados coletados é de extrema importância para assim compreendê-los. Neste sentido, a análise exploratória de dados (AED), visa examinar e adaptar os dados de forma detalhada através de variadas técnicas estatísticas. O Gráfico 1 a seguir ilustra a série de fechamento da bolsa, objeto de estudo, no período anteriormente mencionado (01/01/2019 a 29/11/2024).
    
    **Gráfico 1 - Série Temporal do Preço de Fechamento do Petróleo Brent**
    """)
    # Indicação para imagem: Adicione a imagem do Gráfico 1 aqui
    st.image("imagens/grafico_01_seria_temporal_preco_petroleo.png", caption="Gráfico 1 - Série Temporal do Preço de Fechamento do Petróleo Brent")

    st.write("""
    Em resumo, as medidas descritivas (Tabela 1) mostram que o preço médio do Petróleo Brent é de 73,15 dólares por barril, com valores mínimo e máximo de 19,33 e 127,98 dólares, respectivamente.
    
    **Tabela 1 - Medidas descritivas para a série de preços do Petróleo Brent**
    """)
    # Exemplo de tabela em Streamlit
    data = {
        "Volume de Dados": [1482],
        "Média": [73.15],
        "Desvio Padrão": [19.18],
        "Mínimo": [19.33],
        "Quartil 1": [62.29],
        "Mediana": [74.67],
        "Quartil 3": [84.67],
        "Máximo": [127.98],
        "Coeficiente de Variação": ["26.2%"]
    }
    st.table(pd.DataFrame(data))

    st.write("""
    A dispersão dos dados pode ser considerada moderada se estiver abaixo de 15%, média entre 15% e 30% e alta acima de 30%. Para os dados de preços do Petróleo Brent, a variação é de 26,2%, indicando uma dispersão média dos dados em torno da média. Além disso, um desvio padrão superior indicaria uma maior volatilidade, com oscilações de preços mais acentuadas. Pelas análises descritivas realizadas, percebe-se uma certa variabilidade nos dados, sendo o desvio padrão de 19,18 dólares.
    
    Por meio da combinação do histograma com a curva de densidade (Gráfico 2), é possível visualizar o comportamento dos preços de fechamento do Petróleo Brent. A distribuição apresenta uma forma relativamente normal, o que reafirma visualmente uma certa estabilidade ao longo do período analisado, com variações em torno de um valor médio de 73,15 dólares. A discreta assimetria à direita, observada tanto no histograma quanto no boxplot, pode indicar uma possível tendência de alta a longo prazo.
    
    **Gráfico 2 - Boxplot e Histograma da série do Preço de Fechamento do Petróleo Brent**
    """)
    # Indicação para imagem: Adicione a imagem do Gráfico 2 aqui
    st.image("imagens/grafico_02_box_plot_histograma_preco_petroleo.png", caption="Gráfico 2 - Boxplot e Histograma da série do Preço de Fechamento do Petróleo Brent")

    st.write("""
    Em relação aos outliers, a análise dos dados revela a presença de valores atípicos significativos. No histograma, os outliers são valores discrepantes que se destacam significativamente em relação aos demais e costumam ser identificados como pontos soltos no gráfico. O mesmo comportamento é observado no boxplot, onde há evidências de pontos isolados além das linhas do gráfico de caixa. Especificamente, os preços mínimos de 19,33 dólares e máximos de 127,98 dólares são considerados outliers, indicando a presença de valores discrepantes no fechamento do Petróleo Brent para o período em estudo.
    
    Para compreender a influência dos elementos sazonais e da tendência da série de fechamento, foi realizado a decomposição, mostrada pelo Gráfico 3.
    
    **Gráfico 3 - Decomposição da série do Preço de Fechamento do Preço do Petróleo Brent**
    """)
    # Indicação para imagem: Adicione a imagem do Gráfico 3 aqui
    st.image("imagens/grafico_03_decomposicao_preco_petroleo.png", caption="Gráfico 3 - Decomposição da série do Preço de Fechamento do Preço do Petróleo Brent")

    st.write("""
    A decomposição de uma série temporal, como a do preço de fechamento do Petróleo Brent, busca separar os componentes que a compõem: tendência, sazonalidade e resíduo. Cada componente fornece insights valiosos sobre o comportamento da série:
    
    - **Tendência:** Representa o movimento de longo prazo da série, indicando se há uma tendência de alta, baixa ou estabilidade. No caso do Petróleo Brent, a tendência, quando disponível, pode ajudar a identificar movimentos gerais de crescimento ou declínio no mercado ao longo do tempo.
    - **Sazonalidade:** Refere-se a padrões repetitivos que ocorrem em intervalos regulares de tempo, como variações sazonais anuais ou mensais. No contexto do Petróleo Brent, a sazonalidade captura ciclos curtos e repetitivos que podem refletir mudanças periódicas na demanda ou oferta.
    - **Resíduo:** Corresponde à parte da série que não é explicada pela tendência e sazonalidade, representando o componente aleatório ou "ruído". Resíduos grandes podem indicar anomalias ou eventos significativos no mercado que não foram capturados pelos outros componentes.
    
    Em relação à tendência, cabe ressaltar que a curva apresenta algumas oscilações, mas o movimento de longo prazo pode ser identificado como crescente ou decrescente, dependendo do período analisado. Esse comportamento indica que os preços do Petróleo Brent podem expor um crescimento ou declínio gradual. Ao analisar o componente de sazonalidade, observam-se flutuações significativas que se alteram ao longo do tempo. Embora padrões como o crescimento no início do ano e a alta no início do segundo semestre sejam frequentemente observados, a intensidade desses períodos sazonais não é constante. Variações anuais indicam que fatores externos, como eventos econômicos e geopolíticos, podem influenciar significativamente a magnitude e a direção das mudanças sazonais.
    
    Considerando que para aplicação de alguns modelos de previsão de série temporal, exige-se ao menos que a série seja estacionária, ou seja, a série tem que ter média, variância e covariância finitas e constantes. O teste de hipótese criado por D. A. Dickey e W. A. Fuller e conhecido como teste de Augmented Dickey-Fuller (ADF), tem o intuito de verificar a presença de raiz unitária, ou seja, se a série é estacionária, tendo como hipótese nula a estacionariedade ou a ausência de raiz unitária as hipóteses apresentadas para este teste são:
    
    - **H0:** há uma raiz unitária (ou seja, a série não é estacionária)
    - **H1:** não há uma raiz unitária (ou seja, a série é estacionária)
    
    Ao aplicar o teste ADF aos dados de fechamento do petróleo Brent e considerando um nível de significância de 5%, o p-valor obtido foi de 0.4772. Este valor é maior do que o nível de significância escolhido, o que significa que não se rejeita a hipótese nula. Com um alto grau de confiança, conclui-se que a série apresenta uma raiz unitária e não é estacionária.
    
    **Tabela 2 - Teste de Inferência Estatística para o Preço de Fechamento do Petróleo Brent**
    """)
    # Exemplo de tabela em Streamlit
    data_adf = {
        "Teste": ["Dickey-Fuller (ADF)", "Shapiro-Wilk"],
        "p-valor": [0.4772, 5.419368576173395e-11],
        "Estatística de teste": [-1.6116, 0.98558061585351]
    }
    st.table(pd.DataFrame(data_adf))

    st.write("""
    A normalidade da distribuição possibilita previsões sobre diversos resultados individuais. Desta maneira, faz-se necessário verificar se os dados estão normalmente distribuídos. O teste de hipótese de Shapiro-Wilk calcula uma estatística de teste W e consequentemente averigua se uma determinada amostra aleatória segue uma distribuição normal. As hipóteses adotadas pelo teste são:
    
    - **H0:** As distribuições são normais
    - **H1:** As distribuições não são normais
    
    Para casos em que o valor de p é menor que o nível alfa (significância) escolhido, a hipótese nula é rejeitada e há evidências de que os dados testados não são normalmente distribuídos. O resultado do teste (Tabela 2), atesta a rejeição da hipótese nula, logo, os dados não estão normalmente distribuídos.
    
    A economia global é um sistema interligado, e o mercado de petróleo, representado pelo Preço de Fechamento do Petróleo Brent, reflete essa interconexão. Portanto, foi realizado o estudo de algumas variáveis explicativas, como o volume negociado, que indica o total de contratos de petróleo Brent negociados em um determinado período, além do dólar, utilizado como referência nas transações internacionais e como reserva de valor. A relação dessas variáveis com o preço de fechamento do Petróleo Brent pode ser vista através dos Gráficos 4 e 5.
    
    **Gráfico 4 - Relação do Preço de Fechamento do Petróleo Brent com a variável Volume**
    """)
    # Indicação para imagem: Adicione a imagem do Gráfico 4 aqui
    st.image("imagens/grafico_04_relacao_preco_petroleo_volume.png", caption="Gráfico 4 - Relação do Preço de Fechamento do Petróleo Brent com a variável Volume")

    st.write("""
    **Gráfico 5 - Relação do Preço de Fechamento do Petróleo Brent com a variável Dólar**
    """)
    # Indicação para imagem: Adicione a imagem do Gráfico 5 aqui
    st.image("imagens/grafico_05_relacao_preco_petroleo_dolar.png", caption="Gráfico 5 - Relação do Preço de Fechamento do Petróleo Brent com a variável Dólar")

    st.write("""
    Para o volume negociado, em geral, quando o preço do Petróleo Brent sobe, o volume não necessariamente acompanha o mesmo ritmo. Isso sugere que, em alguns casos, o aumento no preço pode estar associado a uma leve diminuição no volume negociado. Isso pode ocorrer devido a diversos fatores, como tensões geopolíticas, mudanças na oferta e demanda de petróleo, ou variações nas expectativas dos investidores.
    
    O comportamento para o dólar mostra uma similaridade positiva muito fraca em relação ao preço do Petróleo Brent. Essa relação pode ser influenciada por fatores como a política monetária internacional, as taxas de câmbio e o apetite de risco dos investidores, que afetam o fluxo de capitais e a valorização do dólar em relação ao preço do petróleo.
    
    A análise de correlação de Pearson nos fornece uma visão geral das relações lineares entre diferentes variáveis. Esta correlação varia entre -1 e 1, onde um valor próximo de 1 indica uma forte correlação positiva (quando uma variável aumenta, a outra tende a aumentar também), um valor próximo de -1 indica uma forte correlação negativa (quando uma variável aumenta, a outra tende a diminuir), e um valor próximo de 0 indica uma correlação fraca ou nenhuma correlação. Aplicando as variáveis explicativas ao Preço de Fechamento do Petróleo Brent, observamos que o volume negociado apresenta uma correlação fraca de -0.194 com o preço de fechamento, indicando uma leve relação negativa, onde um aumento no volume pode estar levemente associado a uma diminuição no preço. Por outro lado, a variável dólar apresenta uma correlação fraca de 0.072 com o preço de fechamento, sugerindo uma relação positiva muito fraca. A Figura 1, abaixo, expressa a matriz de correlação para as variáveis.
    
    **Figura 1 - Matriz de correlação de Pearson**
    """)
    # Indicação para imagem: Adicione a imagem da Figura 1 aqui
    st.image("imagens/figura_01_matriz_correlacao.png", caption="Figura 1 - Matriz de correlação de Pearson")

elif menu == "Modelo Preditivo":
    st.title("Modelo Preditivo")
    
    st.subheader("Aplicação do Modelo de Previsão")
    
    st.subheader("LSTM (Long Short-Term Memory)")
    st.write("""
    As Redes Neurais Recorrentes (RNNs) representam uma classe poderosa de redes neurais projetadas para processar dados sequenciais, como séries temporais e sequências de texto. Diferente das redes neurais tradicionais, as RNNs possuem conexões recorrentes que permitem a retenção de informações de estados anteriores, tornando-as adequadas para tarefas onde o contexto temporal é crucial.
    
    No entanto, as RNNs tradicionais enfrentam desafios significativos no aprendizado de dependências de longo prazo devido ao problema do desaparecimento e explosão do gradiente, o que limita sua eficácia em muitos cenários práticos. Para superar essas limitações, Sepp Hochreiter e Jürgen Schmidhuber introduziram em 1997 o modelo de Long Short-Term Memory (LSTM).
    
    O LSTM é uma variante das RNNs que incorpora células de memória e mecanismos de portas para gerenciar de forma eficiente a retenção e atualização de informações ao longo do tempo, proporcionando uma solução robusta para o aprendizado de dependências de longo prazo.
    """)

    st.subheader("Estrutura do LSTM")
    st.write("""
    A estrutura do LSTM é composta por:
    
    - **Célula de Memória:** A unidade central que armazena informações ao longo do tempo.
    - **Porta de Entrada:** Decide quais valores das entradas atuais e do estado anterior devem ser atualizados.
    - **Porta de Esquecimento:** Decide quais informações da célula de memória devem ser descartadas.
    - **Porta de Saída:** Decide quais partes da célula de memória serão usadas para produzir a saída.
    
    Devido às suas características, o modelo LSTM é amplamente utilizado para previsão de séries temporais. Desta forma, utilizamos o modelo para realizar a previsão do Preço de Fechamento do Petróleo Brent.
    """)
    # Indicação para imagem: Adicione a imagem da estrutura do LSTM aqui
    st.image("imagens/figura_02_estrutura_lstm.png", caption="Figura 2 - Estrutura LSTM")

    st.subheader("Implementação do LSTM")
    st.write("""
    Para prever o Preço de Fechamento do Petróleo Brent, decidimos não incluir as variáveis exógenas Volume e Dólar no modelo LSTM. A análise de correlação revelou que essas variáveis têm uma relação linear fraca com o preço do petróleo Brent, com correlações negativas e fracas para o volume e positivas, mas igualmente fracas, para o dólar. A inclusão dessas variáveis poderia aumentar a complexidade do modelo sem oferecer melhorias significativas na precisão das previsões. Portanto, focar o modelo LSTM exclusivamente na série temporal dos preços nos permite capturar de forma mais eficaz os padrões e tendências essenciais para previsões precisas.
    
    Essa abordagem se mostrou bastante eficiente. O Gráfico 5 ilustra o desempenho do modelo LSTM, com 80% dos dados usados para treinamento e 20% para teste.
    """)
    # Indicação para imagem: Adicione a imagem do Gráfico 5 aqui
    st.image("imagens/grafico_06_lstm_previsao_preco_petroleo.png", caption="Gráfico 5 - Previsão do Preço de Fechamento do Petróleo Brent usando o Modelo LSTM")

    st.subheader("Avaliação do Desempenho do Modelo")
    st.write("""
    Para avaliar o desempenho do modelo LSTM na previsão do Preço de Fechamento do Petróleo Brent, calculamos as métricas de erro RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error) e R² (Coeficiente de determinação).
    
    **Tabela 3 - Métricas de Erro Modelo de LSTM**
    """)
    # Exemplo de tabela em Streamlit
    data_metrics = {
        "Modelo": ["LSTM"],
        "MAE": [1.316859383372567],
        "RMSE": [1.6208168962291645],
        "MAPE": [0.016422835256464343],
        "R²": ["89.60%"]
    }
    st.table(pd.DataFrame(data_metrics))

    st.write("""
    Os resultados das métricas de avaliação indicam que o modelo LSTM apresenta um desempenho robusto, com baixos valores de MAE, RMSE e MAPE, sugerindo que o modelo está capturando de forma eficaz a dinâmica do preço do petróleo Brent. O coeficiente de determinação (R²) de 89.60% demonstra que o modelo é capaz de explicar uma grande parte da variabilidade nos dados, indicando previsões precisas e confiáveis para este conjunto de dados.
    """)

elif menu == "Dashboard - Exploração e Insights":
    st.title("Dashboard - Exploração e Insights")
    
    st.subheader("Análise Histórica do Preço do Petróleo Brent")
    st.write("""
    Para análise foi considerado os dados históricos do preço do petróleo bruto tipo Brent, considerando dados desde janeiro de 1986 até novembro de 2024.
    
    Com base nesses dados foi desenvolvido um painel de acompanhamento diário para monitoramento das flutuações de preços e busca por eventos que influenciam tais variações.
    """)
    # Indicação para imagem: Adicione a imagem da Figura 3 aqui
    st.image("imagens/figura_03_dashboard_preco_petroleo_1986_2024.png", caption="Figura 3 - Painel da Variação do Preço do Petróleo de 1986 até 2024")

    st.write("""
    A análise histórica dos preços do petróleo é fundamental para compreender as dinâmicas do mercado global e a influência de eventos econômicos, políticos e sociais sobre o setor energético. O preço do barril de petróleo Brent, referência mundial, é amplamente impactado por acontecimentos como crises financeiras, conflitos geopolíticos e alterações na demanda global, refletindo a sensibilidade do mercado a fatores externos. Eventos como a pandemia de Covid-19, a Crise Financeira de 2008, os ataques de 11 de setembro de 2001 e a invasão do Kuwait em 1990 são exemplos de como o cenário global pode gerar flutuações severas nos preços. Essas análises permitem não apenas compreender as tendências passadas, mas também preparar estratégias mais eficazes para lidar com os desafios e oportunidades do futuro.
    """)

    st.subheader("Eventos")
    
    st.subheader("Pandemia Covid-19")
    st.write("""
    **Período de Março de 2020 a Maio de 2023 (marco oficial da OMS).**
    
    A pandemia de Covid-19 causou um impacto sem precedentes nos preços do petróleo, especialmente durante o primeiro semestre de 2020. Com a implementação de lockdowns em diversos países, as restrições à mobilidade e o fechamento de indústrias reduziram drasticamente a demanda global por petróleo. Em abril de 2020, a situação atingiu um marco histórico quando os contratos futuros do WTI chegaram a valores negativos e o Brent caiu abaixo de US$ 20 por barril, reflexo do excesso de oferta e da falta de capacidade de armazenamento. Esse colapso destacou a vulnerabilidade do mercado a mudanças abruptas na demanda. Contudo, a partir do segundo semestre de 2020, com a flexibilização das restrições, estímulos econômicos e o início da vacinação em massa, a recuperação econômica global impulsionou a retomada da demanda energética. Essa recuperação fez os preços do Brent subirem consistentemente, ultrapassando os US$ 50 por barril no final do ano, evidenciando a resiliência do mercado mesmo diante de crises severas.
    """)
    # Indicação para imagem: Adicione a imagem da Figura 4 aqui
    st.image("imagens/figura_04_dashboard_preco_petroleo_covid-19.png", caption="Figura 4 - Painel da Variação do Preço do Petróleo - Pandemia Covid 19")

    st.subheader("Crise Financeira Global")
    st.write("""
    **Período de Setembro de 2008 a Março de 2009.**
    
    Outro evento marcante foi a Crise Financeira Global de 2008-2009, que causou uma queda drástica no preço do barril de petróleo Brent. Durante esse período, iniciado com o colapso do Lehman Brothers em setembro de 2008, a recessão global resultante reduziu significativamente a demanda por petróleo. Os preços, que estavam em torno de US$ 140 por barril em julho de 2008, caíram para cerca de US$ 40 por barril em dezembro do mesmo ano.
    
    A recuperação começou gradualmente no início de 2009, acompanhando os estímulos econômicos e a estabilização dos mercados financeiros.
    """)
    # Indicação para imagem: Adicione a imagem da Figura 5 aqui
    st.image("imagens/figura_05_dashboard_preco_petroleo_crise_financeira_global.png", caption="Figura 5 - Painel da Variação do Preço do Petróleo - Crise Financeira Global")

    st.subheader("Ataques de 11 Setembro e Guerra ao Terror")
    st.write("""
    **Período de Setembro de 2001 a Dezembro de 2003.**
    
    Os Ataques de 11 de Setembro de 2001 e a subsequente Guerra ao Terror (2001-2003) também desempenharam um papel significativo nas variações do preço do petróleo Brent. Após os ataques, o mercado global foi tomado por incertezas, e os preços oscilaram com a crescente percepção de risco geopolítico, especialmente devido ao papel estratégico do Oriente Médio na produção de petróleo.
    
    A escalada para a Guerra no Iraque em 2003 intensificou essas preocupações, levando a picos nos preços. Durante este período, as interrupções no fornecimento e o aumento da volatilidade marcaram o mercado. O preço do barril subiu de cerca de US$ 20 por barril no início de 2002 para mais de US$ 30 em meados de 2003, refletindo o impacto da instabilidade geopolítica.
    """)
    # Indicação para imagem: Adicione a imagem da Figura 6 aqui
    st.image("imagens/figura_06_dashboard_preco_petroleo_11_setembro.png", caption="Figura 6 - Painel da Variação do Preço do Petróleo - Ataques de 11 Setembro e Guerra ao Terror")

    st.subheader("Invasão do Kuwait pelo Iraque")
    st.write("""
    **Período de Agosto de 1990 a Fevereiro de 1991.**
    
    A Invasão do Kuwait pelo Iraque, ocorrida de agosto de 1990 a fevereiro de 1991, foi um evento crucial para as severas variações nos preços do petróleo Brent. Esse conflito, que culminou na Guerra do Golfo, interrompeu significativamente a produção de petróleo do Kuwait e gerou temores sobre a estabilidade de outras importantes regiões produtoras no Oriente Médio.
    
    Durante o auge da crise, os preços do petróleo dispararam de cerca de US$ 18 por barril em julho de 1990 para mais de US$ 40 por barril em outubro do mesmo ano, marcando um dos períodos de maior alta em curto prazo na história do mercado petrolífero. Com o fim do conflito em fevereiro de 1991 e a retomada da produção, os preços começaram a se estabilizar.
    """)
    # Indicação para imagem: Adicione a imagem da Figura 7 aqui
    st.image("imagens/figura_07_dashboard_preco_petroleo_invasao_kwait.png", caption="Figura 7 - Painel da Variação do Preço do Petróleo - Invasão do Kuwait pelo Iraque")

    st.write("""
    Esses quatro eventos fornecem uma base sólida para entender como fatores globais afetam o mercado de petróleo.
    """)

elif menu == "MVP e Plano de Deploy":
    st.title("MVP e Plano de Deploy do Modelo")
    
    st.subheader("Implementação do MVP")
    st.write("""
    Para disponibilizar o modelo preditivo do Preço de Fechamento do Petróleo Brent de forma prática e acessível, desenvolvemos um MVP utilizando Streamlit e planejamos um deploy simplificado que possibilita o uso imediato do modelo em um ambiente de produção.
    
    O MVP (Minimum Viable Product) foi criado para permitir a interação com o modelo preditivo de maneira intuitiva e eficaz. Utilizamos Streamlit por sua capacidade de transformar scripts de dados em aplicativos web interativos rapidamente. As principais funcionalidades do MVP incluem:
    
    - **Interface Intuitiva:** Design acessível para usuários sem conhecimento técnico.
    - **Entrada de Dados:** Possibilidade de simular cenários futuros.
    - **Visualização de Resultados:** Gráfico que apresenta previsões dos preços com horizonte de previsão definido para até 15 dias.
    """)

    st.subheader("Plano de Deploy Simplificado")
    st.write("""
    Para operacionalizar o MVP e garantir que o modelo esteja disponível para uso contínuo, seguimos um plano de deploy simplificado:
    
    1. **Desenvolvimento em Ambiente Local:**
       - Criamos e testamos o modelo preditivo em um ambiente local, garantindo que todas as funcionalidades estejam operacionais e que o modelo esteja devidamente treinado.
       - Salvamos o modelo no formato .keras, desta forma, asseguramos que todos os detalhes do modelo, incluindo a arquitetura, os pesos e a configuração de treinamento, sejam preservados.
       - Criamos e testamos a aplicação (MVP).
       
    2. **Código no GitHub:**
       - Subimos o código do modelo, a aplicação e os arquivos auxiliares do modelo e dos dados para um repositório no GitHub, facilitando o controle de versão e o acesso ao código-fonte por toda a equipe.
       
    3. **Configuração do Streamlit:**
       - Criamos uma conta na Streamlit Cloud (streamlit.io). Esse passo é executado apenas uma vez.
       - Configuramos o aplicativo para consumir o modelo diretamente do repositório no GitHub. Streamlit oferece uma integração direta com GitHub, simplificando o processo de deploy. Esse passo também é executado apenas uma vez, mas pode ser ajustado conforme necessário.
       
    4. **Deploy e Acesso:**
       - Com a configuração concluída, o aplicativo Streamlit foi implantado diretamente na nuvem, tornando-o acessível a qualquer usuário com acesso à internet. Isso permite que o modelo preditivo seja utilizado sem a necessidade de infraestrutura complexa ou manutenção contínua.
    
    Este plano de deploy simplificado garante que o modelo preditivo esteja disponível de forma rápida e eficiente, permitindo que usuários interajam com o modelo e obtenham previsões precisas sem complicações técnicas adicionais.
    """)

elif menu == "Previsão do Preço do Petróleo":
    st.title("Previsão do Preço do Petróleo Brent")
    st.subheader("Previsão do Preço do Petróleo Brent")
    st.write("""
    Para obter a previsão do Preço de Fechamento do Petróleo Brent, por favor, selecione a data desejada. Para assegurar a máxima precisão das previsões, a escolha da data está limitada a um horizonte de até 15 dias a partir da data atual.
    """)

    # Entrada de data do usuário com limite de data máxima
    input_date = st.date_input(
        "Selecione uma data para a previsão:",
        date.today(),
        min_value=last_date + timedelta(days=1),
        max_value=max_forecast_date
    )

    # Botão para fazer a previsão
    if st.button("Prever"):
        # Gerar previsões a partir dos últimos 'NUM_DAYS' de dados históricos até a data selecionada
        predictions = generate_predictions(df_close, model, scaler, input_date, NUM_DAYS)
        
        # Criar DataFrame para exibir as previsões
        df_predictions = pd.DataFrame(predictions, columns=['Data', 'Preço'])
        
        # Converter a coluna de data para datetime
        df_predictions['Data'] = pd.to_datetime(df_predictions['Data'])
        
        # Incluir o último ponto dos dados históricos nos dados preditos
        last_historical_point = df_close.iloc[-1]
        df_predictions = pd.concat([
            pd.DataFrame({'Data': [last_historical_point['Date']], 'Preço': [last_historical_point['Close']]}),
            df_predictions
        ]).reset_index(drop=True)
        
        # Filtrar os últimos 'NUM_DAYS' de dados históricos
        df_close_last_n = df_close.tail(NUM_DAYS)
        
        fig = go.Figure()
        
        # Dados históricos e preenchidos
        fig.add_trace(go.Scatter(x=df_close_last_n['Date'], y=df_close_last_n['Close'],
                                 mode='lines+markers', name='Dados Históricos', line=dict(color='blue')))
        
        # Dados preditos
        fig.add_trace(go.Scatter(x=df_predictions['Data'], y=df_predictions['Preço'],
                                 mode='lines+markers', name='Previsão', line=dict(color='lightblue')))
        
        # Adicionar layout ao gráfico
        fig.update_layout(title='Previsão do Preço de Fechamento do Petróleo Brent',
                          xaxis_title='', yaxis_title='Preço do Petróleo',
                          legend=dict(x=0, y=1), hovermode='x unified')
        
        # Renderizar o gráfico no Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Exibir a tabela de previsões
        st.subheader("Previsões de Preços por Data")
        st.write(df_predictions)

elif menu == "Conclusão":
    st.title("Conclusão")
    st.write("""
    O projeto desenvolvido para a consultoria teve como foco principal a análise e previsão dos preços do petróleo Brent, utilizando dados históricos obtidos do site do biblioteca yfinance, disponibilizada pelo portal Yahoo! Finance. Através da implementação de um modelo de Machine Learning, especificamente um LSTM (Long Short-Term Memory), e a criação de um dashboard interativo, conseguimos atender aos requisitos do cliente, oferecendo insights valiosos para a tomada de decisão estratégica.

    O desenvolvimento de um modelo preditivo para o Preço de Fechamento do Petróleo Brent, utilizando a arquitetura LSTM, demonstrou ser uma abordagem eficaz para capturar a complexidade inerente às séries temporais deste mercado. A escolha de não incluir variáveis exógenas, como Volume e Dólar, foi justificada pela análise de correlação, que indicou uma relação linear fraca entre essas variáveis e o preço do petróleo. Essa decisão simplificou o modelo e permitiu um foco mais preciso nos padrões temporais, resultando em previsões robustas com um coeficiente de determinação (R²) de 89.60%.

    A análise exploratória dos dados e a decomposição da série temporal forneceram insights valiosos sobre as tendências e sazonalidades do mercado, enquanto a avaliação das métricas de erro, como MAE, RMSE e MAPE, confirmou a eficácia do modelo em prever variações de preços com precisão.

    Além disso, o desenvolvimento de um dashboard interativo em Power BI complementou a análise, oferecendo uma visualização abrangente das flutuações de preços e dos fatores externos que as influenciam, como eventos geopolíticos e crises econômicas. Este dashboard é uma ferramenta importante para a tomada de decisões informadas, permitindo que os usuários explorem dados históricos e identifiquem padrões e anomalias de forma intuitiva.

    A análise histórica revelou quatro insights principais, incluindo o impacto da pandemia de Covid-19, a Crise Financeira Global de 2008, os Ataques de 11 de Setembro e a Invasão do Kuwait pelo Iraque. Esses eventos demonstraram a sensibilidade do mercado de petróleo a fatores externos e a importância de uma análise contextualizada para prever tendências futuras.

    O plano de deploy simplificado, utilizando Streamlit, permitiu a criação de um MVP funcional que facilita a interação com o modelo preditivo em um ambiente de produção. Este processo garantiu que o modelo estivesse disponível de forma rápida e eficiente, sem a necessidade de infraestrutura complexa.

    Em resumo, este projeto não apenas alcançou seu objetivo de prever o preço do petróleo Brent com alta precisão, mas também forneceu uma plataforma analítica poderosa para explorar e compreender as dinâmicas do mercado de petróleo. Os insights gerados por este modelo podem ser fundamentais para estratégias de investimento e gestão de risco, destacando a importância de tecnologias avançadas de Machine Learning na análise de mercados complexos e voláteis.

    A conclusão bem-sucedida deste projeto não apenas atende às necessidades imediatas do cliente, mas também estabelece uma base sólida para futuras análises e desenvolvimentos. A capacidade de prever com precisão os preços do petróleo Brent pode auxiliar na formulação de estratégias de mercado mais informadas e na mitigação de riscos associados à volatilidade dos preços.

    Para melhorias futuras, recomenda-se explorar a inclusão de variáveis exógenas adicionais que possam enriquecer o modelo preditivo, como indicadores econômicos mais amplos ou dados climáticos. Além disso, a expansão das funcionalidades do dashboard para incluir análises preditivas em tempo real poderia aumentar ainda mais o valor estratégico oferecido aos usuários.

    """)
