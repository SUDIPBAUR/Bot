import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import os

# Binance setup with error handling
def setup_exchange():
    api_key = "zSytUpX7UqgSZH2Ig9DrTpEQTR77wBqTwldiLIrwoJWpXUU1ksUAywdxUWVMtCsm"
    api_secret = "orEZ15ayLiAfBjwYuEDLlTsNwrp4hIec8GSyWvOqmEbImpzsjJU9PmKbpxRqijzu"
    
    if not api_key or not api_secret:
        st.error("Binance API keys not found in environment variables!")
        st.stop()
    
    return ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True}
    })

exchange = setup_exchange()

def fetch_data():
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')
    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        st.stop()

def add_indicators(df):
    try:
        # Corrected RSI calculation
        df['rsi'] = RSIIndicator(df['close']).rsi()
        
        # Other indicators
        df['macd'] = MACD(df['close']).macd()
        df['ema'] = EMAIndicator(df['close']).ema_indicator()
        df['stoch'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        bb = BollingerBands(df['close'])
        df['bb_bbm'] = bb.bollinger_mavg()
        df['bb_bbh'] = bb.bollinger_hband()
        df['bb_bbl'] = bb.bollinger_lband()
        
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        return df.dropna()
    except Exception as e:
        st.error(f"Indicator error: {str(e)}")
        st.stop()

def prepare_data(df):
    df['target'] = df['close'].shift(-1)
    features = ['close', 'rsi', 'macd', 'ema', 'stoch', 'cci', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'obv']
    data = df[features].values
    target = df['target'].values

    sequence_length = 20
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(1 if target[i+sequence_length] > data[i+sequence_length-1][0] else 0)

    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Database simulation
signals_db = []
def insert_signal(timestamp, signal, confidence, price):
    signals_db.append({
        'timestamp': timestamp,
        'signal': signal,
        'confidence': confidence,
        'price': price
    })

def fetch_signals():
    return signals_db[-10:]  # Return last 10 signals

def place_order(signal):
    return f"Simulated {signal} order executed at {datetime.datetime.now()}"

def main():
    st.title("BTC/USDT AI Trading Bot")
    
    try:
        df = fetch_data()
        df = add_indicators(df)
        X, y = prepare_data(df)
        
        if len(X) < 20:
            st.error("Not enough data for training")
            return
            
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X[:-10], y[:-10], epochs=5, batch_size=32, verbose=0)
        
        prediction = model.predict(X[-1:])[0][0]
        signal = 'BUY' if prediction > 0.6 else 'SELL' if prediction < 0.4 else 'HOLD'
        
        insert_signal(datetime.datetime.now(), signal, prediction, df['close'].iloc[-1])
        
        # Display results
        st.line_chart(df['close'])
        st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
        st.subheader(f"Signal: {signal} (Confidence: {prediction:.2%})")
        
        if st.button("Execute Trade"):
            result = place_order(signal)
            st.success(result)
            
        st.subheader("Recent Signals")
        st.table(pd.DataFrame(fetch_signals()))
        
    except Exception as e:
        st.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()
