import streamlit as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

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

# Configurar a interface do Streamlit
st.title("Previsão do Preço de Fechamento do Petróleo Brent")
st.write("Informe uma data futura para prever o Preço de Fechamento do Petróleo Brent.")

# Entrada de data do usuário com limite de data máxima
input_date = st.date_input(
    "Selecione uma data futura",
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
    
    # Visualização de Resultados: Gráfico Interativo com Plotly
    st.subheader("Gráfico de Previsões Interativo")
    
    # Filtrar os últimos 'NUM_DAYS' de dados históricos
    df_close_last_n = df_close.tail(NUM_DAYS)
    
    fig = go.Figure()
    
    # Dados históricos e preenchidos
    fig.add_trace(go.Scatter(x=df_close_last_n['Date'], y=df_close_last_n['Close'],
                             mode='lines+markers', name='Histórico e Preenchido', line=dict(color='blue')))
    
    # Dados preditos
    fig.add_trace(go.Scatter(x=df_predictions['Data'], y=df_predictions['Preço'],
                             mode='lines+markers', name='Previsão', line=dict(color='lightblue')))
    
    # Adicionar layout ao gráfico
    fig.update_layout(title='Previsão do Preço de Fechamento do Petróleo Brent',
                      xaxis_title='Data', yaxis_title='Preço',
                      legend=dict(x=0, y=1), hovermode='x unified')
    
    # Renderizar o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Exibir a tabela de previsões
    st.subheader("Previsões de Preços")
    st.write(df_predictions)
