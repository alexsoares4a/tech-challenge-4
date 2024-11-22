import streamlit as st
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Carregar o modelo LSTM salvo
model = load_model('lstm_model.keras')

# Carregar os dados do CSV para ajustar o scaler
df_close = pd.read_csv('dados_petroleo.csv')
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_close[['Close']])

# Função para verificar se uma data é um dia útil
def is_weekday(date):
    return date.weekday() < 5  # 0 é segunda-feira, 4 é sexta-feira

# Função para gerar previsões para datas úteis
def generate_predictions(start_date, end_date):
    current_date = start_date
    predictions = []
    
    # Prepare uma sequência de dados para prever
    last_days = df_close[['Close']].values[-20:]  # Pega os últimos 20 dias
    last_days_scaled = scaler.transform(last_days)
    
    while current_date <= end_date:
        if is_weekday(current_date):
            # Adicionar a previsão ao final da sequência
            date_array_scaled = np.array([last_days_scaled])
            
            # Fazer a previsão
            prediction_scaled = model.predict(date_array_scaled)
            
            # Reverter a normalização
            prediction = scaler.inverse_transform(
                np.concatenate((prediction_scaled, np.zeros((prediction_scaled.shape[0], df_close.shape[1] - 1))), axis=1)
            )[:, 0]
            
            predictions.append((current_date, prediction[0]))
            
            # Atualizar a sequência de dados para a próxima previsão
            last_days_scaled = np.append(last_days_scaled[1:], prediction_scaled, axis=0)
        
        current_date += timedelta(days=1)
    
    return predictions

# Configurar a interface do Streamlit
st.title("Previsão do Preço de Fechamento do Petróleo Brent")
st.write("Informe uma data futura para prever o Preço de Fechamento do Petróleo Brent.")

# Entrada de data do usuário
input_date = st.date_input("Selecione uma data futura", date.today())

# Botão para fazer a previsão
if st.button("Prever"):
    # Gerar previsões para datas úteis entre hoje e a data selecionada
    today = date.today()
    predictions = generate_predictions(today + timedelta(days=1), input_date)
    
    # Criar DataFrame para exibir as previsões
    df_predictions = pd.DataFrame(predictions, columns=['Data', 'Preço'])
    
    # Exibir a tabela de previsões
    st.write(df_predictions)
