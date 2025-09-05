import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
from pycoingecko import CoinGeckoAPI

st.set_page_config(page_title="Radar CoinGecko — Comprar/Vender (BRL ou USD)", layout="wide")
st.title("📡 Radar — Melhores para **Comprar** e **Vender** agora (fonte: CoinGecko)")
st.caption("Sem chave de API. Sinais: Breakout + Volume + Tendência (VWAP/EMA20) + RSI. Níveis com ATR.")

cg = CoinGeckoAPI()

# ---------- Helpers ----------
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def vwap(close, volume, length=20):
    return (close * volume).rolling(length).sum() / volume.rolling(length).sum()

def atr_from_hlc(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_indicators(df, lookback, vol_mult, rsi_thr):
    d = df.copy()
    d["open"] = d["close"].shift(1).fillna(d["close"])
    d["high"] = d["close"].rolling(3).max()
    d["low"]  = d["close"].rolling(3).min()

    d["EMA20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["RSI"] = rsi(d["close"], 14)
    d["VWAP"] = vwap(d["close"], d["volume"], 20)
    d["ATR"] = atr_from_hlc(d["high"], d["low"], d["close"], 14)

    roll_high = d["high"].shift(1).rolling(lookback).max()
    roll_low  = d["low"].shift(1).rolling(lookback).min()
    vol_ok = d["volume"] > (d["volume"].rolling(lookback).mean() * vol_mult)
    trend_ok = (d["close"] > d["VWAP"]) & (d["close"] > d["EMA20"])
    rsi_ok = d["RSI"] > rsi_thr

    breakout_up = d["close"] > roll_high
    breakout_down = d["close"] < roll_low

    d["BUY_SIGNAL"] = breakout_up & vol_ok & trend_ok & rsi_ok
    d["SELL_SIGNAL"] = breakout_down | ((d["close"] < d["EMA20"]) & (d["close"] < d["VWAP"]) & (d["RSI"] < 50) & vol_ok)
    return d

@st.cache_data(ttl=120, show_spinner=False)
def get_top_coins(vs_currency="brl", per_page=50):
    data = cg.get_coins_markets(vs_currency=vs_currency, per_page=per_page, page=1, price_change_percentage="1h,24h,7d")
    return pd.DataFrame(data)[["id","symbol","name","current_price","total_volume","price_change_percentage_24h"]]

@st.cache_data(ttl=90, show_spinner=False)
def get_minute_chart(coin_id, vs_currency="brl", days=1):
    chart = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days, interval="minute")
    prices = pd.DataFrame(chart["prices"], columns=["ts","price"])
    vols = pd.DataFrame(chart["total_volumes"], columns=["ts","volume"])
    df = prices.merge(vols, on="ts")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("timestamp").drop(columns=["ts"])
    df = df.rename(columns={"price":"close"})
    df["volume"] = df["volume"].diff().clip(lower=0).fillna(0)
    return df.tail(400)

def decision_from_row(last, atr_mult_sl, atr_mult_tp, rsi_thr):
    price = float(last["close"])
    atr = float(last["ATR"]) if np.isfinite(last["ATR"]) else 0.0
    atr = atr if atr > 0 else price * 0.002
    sl = price - atr_mult_sl * atr
    tp = price + atr_mult_tp * atr
    if bool(last["BUY_SIGNAL"]):
        return "COMPRAR", price, sl, tp
    if bool(last["SELL_SIGNAL"]):
        return "VENDER/REALIZAR", price, price + atr_mult_sl * atr, price - atr_mult_tp * atr
    return "ESPERAR", price, sl, tp

# ---------- Controls ----------
colA, colB, colC = st.columns([1,1,1])
fiat = colA.selectbox("Fiat (vs_currency)", ["brl", "usd"], index=0)
timeframe_hint = colB.selectbox("Timeframe alvo", ["15m", "5m", "1h"], index=0)
universe_size = int(colC.number_input("Tamanho do universo (top por market cap)", min_value=10, max_value=200, value=50, step=10))

lookback = st.slider("Lookback do breakout (barras)", 10, 50, 20, 1)
vol_mult = st.slider("Múltiplo de volume vs. média", 1.0, 3.0, 1.6, 0.1)
rsi_thr = st.slider("RSI mínimo", 40, 60, 50, 1)
atr_mult_sl = st.slider("ATR p/ Stop (x)", 0.5, 2.0, 1.0, 0.1)
atr_mult_tp = st.slider("ATR p/ Take (x)", 1.5, 5.0, 3.0, 0.5)

st.write("🔎 Carregando universo do CoinGecko…")
coins = get_top_coins(vs_currency=fiat, per_page=universe_size)

if coins.empty:
    st.error("Não foi possível carregar as moedas do CoinGecko.")
    st.stop()

st.success(f"Universo carregado: {len(coins)} moedas ({fiat.upper()}).")

rows_buy, rows_sell = [], []
progress = st.progress(0, text="Varrendo séries intradiárias…")

for i, row in enumerate(coins.itertuples(index=False), start=1):
    try:
        df = get_minute_chart(row.id, vs_currency=fiat, days=1)
        if df.empty or len(df) < 60:
            continue
        d = compute_indicators(df, lookback, vol_mult, rsi_thr)
        last = d.iloc[-1]
        action, price, sl, tp = decision_from_row(last, atr_mult_sl, atr_mult_tp, rsi_thr)
        score = 0
        if action == "COMPRAR":
            score += 2
            if last["close"] > last["VWAP"]: score += 1
            if last["RSI"] > (rsi_thr + 5): score += 1
            rr = (tp - price) / max(price - sl, 1e-9)
            rows_buy.append([row.name, row.symbol.upper(), row.id, price, sl, tp, float(last["RSI"]), float(last["VWAP"]), float(last["ATR"]), round(rr,2), score])
        elif action == "VENDER/REALIZAR":
            score += 2
            if last["close"] < last["VWAP"]: score += 1
            if last["RSI"] < 45: score += 1
            rows_sell.append([row.name, row.symbol.upper(), row.id, price, float(last["RSI"]), float(last["VWAP"]), float(last["ATR"]), score])
    except Exception:
        pass
    progress.progress(i/len(coins), text=f"Varrendo… {i}/{len(coins)}")

progress.empty()

def fmt(x): return f"{x:,.6f}".replace(",", "X").replace(".", ",").replace("X",".")

if rows_buy:
    df_buy = pd.DataFrame(rows_buy, columns=["Nome","Símbolo","id","Preço","SL","TP","RSI","VWAP","ATR","R:R","Score"])\
        .sort_values(["Score","R:R","RSI"], ascending=[False,False,False]).head(20)
    for c in ["Preço","SL","TP","VWAP","ATR"]: df_buy[c] = df_buy[c].map(fmt)
    st.subheader("🟢 Top **COMPRAR** agora")
    st.dataframe(df_buy.reset_index(drop=True), use_container_width=True)
else:
    st.info("Sem sinais fortes de COMPRA agora.")

if rows_sell:
    df_sell = pd.DataFrame(rows_sell, columns=["Nome","Símbolo","id","Preço","RSI","VWAP","ATR","Score"])\
        .sort_values(["Score","RSI"], ascending=[False,True]).head(20)
    for c in ["Preço","VWAP","ATR"]: df_sell[c] = df_sell[c].map(fmt)
    st.subheader("🔴 Top **VENDER / REALIZAR** agora")
    st.dataframe(df_sell.reset_index(drop=True), use_container_width=True)
else:
    st.info("Sem sinais fortes de VENDA agora.")

st.caption(f"Atualizado: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC — Fonte: CoinGecko")
