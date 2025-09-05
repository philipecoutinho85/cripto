
import time
import pandas as pd
import numpy as np
import streamlit as st

# ---- Lightweight ccxt import (installed via requirements.txt) ----
import ccxt

st.set_page_config(page_title="Sinal Cripto — Comprar ou Vender", layout="centered")

st.title("🟢🔴 Sinal Cripto — Comprar ou Vender")
st.caption("Regras simples: Breakout + Volume + Tendência. SL/TP por ATR.")

# ---------------- Helpers ----------------
def rsi(series, period=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(period).mean()
    roll_down = pd.Series(down, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df, period=14):
    return true_range(df).rolling(period).mean()

def vwap(df, length=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    return (tp * vol).rolling(length).sum() / vol.rolling(length).sum()

# ---------------- UI ----------------
symbols_default = ["ADA/USDT", "SOL/USDT", "DOGE/USDT", "ETH/USDT", "BTC/USDT"]
col1, col2 = st.columns(2)
symbol = col1.selectbox("Par (Binance Spot)", symbols_default, index=0)
timeframe = col2.selectbox("Timeframe", ["15m", "5m", "1h"], index=0)
lookback = st.slider("Lookback do breakout (barras)", 10, 50, 20, 1)
vol_mult = st.slider("Múltiplo de volume (x média)", 1.0, 3.0, 1.5, 0.1)
rsi_thr = st.slider("RSI mínimo", 40, 60, 50, 1)
atr_mult_sl = st.slider("ATR para Stop (x)", 0.5, 2.0, 1.0, 0.1)
atr_mult_tp = st.slider("ATR para Take (x)", 1.5, 5.0, 3.0, 0.5)

with st.expander("Opcional: informe sua entrada para ver SL/TP e ação de saída"):
    entry_price = st.number_input("Preço de entrada (deixe 0 se ainda não entrou)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")

# ---------------- Data ----------------
st.write("Buscando dados da Binance…")
ex = ccxt.binance()
limit = 400
ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)

# Indicators
df["RSI"] = rsi(df["close"], 14)
df["ATR"] = atr(df, 14)
df["VWAP"] = vwap(df, 20)
df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()

# Signals
roll_high = df["high"].shift(1).rolling(lookback).max()
vol_ok = df["volume"] > (df["volume"].rolling(lookback).mean() * vol_mult)
trend_ok = (df["close"] > df["VWAP"]) & (df["close"] > df["EMA20"])
rsi_ok = df["RSI"] > rsi_thr
breakout = df["close"] > roll_high

df["BUY_SIGNAL"] = breakout & vol_ok & trend_ok & rsi_ok

last = df.iloc[-1]

price = float(last["close"])
atr_val = float(last["ATR"])
sl = price - atr_mult_sl * atr_val
tp = price + atr_mult_tp * atr_val

st.metric("Preço atual", f"{price:,.6f}".replace(",", "X").replace(".", ",").replace("X","."))
st.write(f"RSI: **{last['RSI']:.1f}** | ATR: **{atr_val:.6f}** | VWAP: **{last['VWAP']:.6f}** | EMA20: **{last['EMA20']:.6f}**")

# ---------------- Decision Logic ----------------
action = "ESPERAR"
reason = "Condições incompletas."
color = "gray"

if bool(last["BUY_SIGNAL"]):
    action = "COMPRAR"
    reason = f"Breakout + Volume ({vol_mult}x) + Tendência (acima do VWAP/EMA20) + RSI>{rsi_thr}"
    color = "green"
else:
    # if in position (entry provided), decide hold/sell
    if entry_price > 0.0:
        sl_from_entry = entry_price - atr_mult_sl * atr_val
        tp_from_entry = entry_price + atr_mult_tp * atr_val
        if price <= sl_from_entry:
            action = "VENDER"
            reason = "Preço atingiu Stop Loss calculado por ATR."
            color = "red"
        elif price >= tp_from_entry:
            action = "REALIZAR LUCRO"
            reason = "Preço atingiu Take Profit calculado por ATR."
            color = "green"
        elif price > entry_price and price > last["VWAP"]:
            action = "MANTER"
            reason = "Acima da entrada e do VWAP — manter até TP ou trailing."
            color = "blue"
        else:
            action = "ESPERAR"
            reason = "Sem breakout válido. Aguardar."
            color = "gray"

# Big badge (use triple single quotes inside code to avoid conflict in outer string)
st.markdown(f'''
<div style="text-align:center; padding:16px; border-radius:12px; background:#f6f6f6; border:1px solid #ddd;">
  <div style="font-size:48px; font-weight:800; color:{{'green' if color=='green' else ('#d00' if color=='red' else ('#06c' if color=='blue' else '#666'))}};">
    {action}
  </div>
  <div style="margin-top:6px; font-size:14px;">{reason}</div>
</div>
''', unsafe_allow_html=True)

# SL/TP suggestions (from current price)
st.subheader("Níveis sugeridos (baseados no ATR)")
colA, colB = st.columns(2)
with colA:
    st.write("**Stop Loss (SL)**")
    st.code(f"{sl:.6f}")
with colB:
    st.write("**Take Profit (TP)**")
    st.code(f"{tp:.6f}")

st.caption("Dica: use ordens OCO/TP/SL na exchange para automatizar sua saída.")
