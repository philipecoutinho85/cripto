
import time
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import ccxt

st.set_page_config(page_title="Radar Binance — Melhores para Comprar/Vender", layout="wide")

st.title("📡 Radar Binance — Melhores para **Comprar** e **Vender** agora")
st.caption("Sinais simples e objetivos: Breakout + Volume + Tendência (VWAP/EMA20) + RSI. Níveis de SL/TP por ATR.")

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

@st.cache_data(ttl=120, show_spinner=False)
def list_symbols(exchange_id="binance", quote="BRL"):
    ex = ccxt.binance()
    markets = ex.load_markets()
    syms = []
    for s, m in markets.items():
        if m.get("spot") and m.get("active", True):
            base = m.get("base", "")
            q = m.get("quote", "")
            if q == quote:
                syms.append(s)
    return sorted(list(set(syms)))

@st.cache_data(ttl=60, show_spinner=False)
def fetch_tickers_dict(quote="BRL"):
    ex = ccxt.binance()
    tickers = ex.fetch_tickers()
    # Return dict filtered to quote
    out = {}
    for sym, t in tickers.items():
        if sym.endswith("/" + quote):
            out[sym] = t
    return out

@st.cache_data(ttl=60, show_spinner=False)
def get_top_by_volume(quote="BRL", topn=50):
    t = fetch_tickers_dict(quote)
    if not t:
        return []
    # ccxt tickers may include 'quoteVolume' or 'baseVolume'
    rows = []
    for sym, d in t.items():
        qv = d.get("quoteVolume") or 0.0
        bv = d.get("baseVolume") or 0.0
        rows.append((sym, float(qv), float(bv)))
    df = pd.DataFrame(rows, columns=["symbol","quoteVolume","baseVolume"]).sort_values("quoteVolume", ascending=False)
    return df["symbol"].head(topn).tolist()

@st.cache_data(ttl=90, show_spinner=False)
def load_ohlcv(symbol, timeframe="15m", limit=300):
    ex = ccxt.binance()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def compute_signals(df, lookback, vol_mult, rsi_thr):
    d = df.copy()
    d["RSI"] = rsi(d["close"], 14)
    d["ATR"] = atr(d, 14)
    d["VWAP"] = vwap(d, 20)
    d["EMA20"] = d["close"].ewm(span=20, adjust=False).mean()

    roll_high = d["high"].shift(1).rolling(lookback).max()
    roll_low  = d["low"].shift(1).rolling(lookback).min()

    vol_ok = d["volume"] > (d["volume"].rolling(lookback).mean() * vol_mult)
    trend_ok = (d["close"] > d["VWAP"]) & (d["close"] > d["EMA20"])
    rsi_ok = d["RSI"] > rsi_thr

    breakout_up = d["close"] > roll_high
    breakout_down = d["close"] < roll_low

    d["BUY_SIGNAL"] = breakout_up & vol_ok & trend_ok & rsi_ok
    # SELL / realizar: perda de força — fecha abaixo de EMA20 e abaixo do VWAP com RSI<50 e volume alto
    d["SELL_SIGNAL"] = breakout_down | ((d["close"] < d["EMA20"]) & (d["close"] < d["VWAP"]) & (d["RSI"] < 50) & vol_ok)
    return d

def make_decision(last_row, atr_mult_sl, atr_mult_tp):
    price = float(last_row["close"])
    atr_val = float(last_row["ATR"])
    sl = price - atr_mult_sl * atr_val
    tp = price + atr_mult_tp * atr_val
    if bool(last_row.get("BUY_SIGNAL", False)):
        return "COMPRAR", price, sl, tp
    if bool(last_row.get("SELL_SIGNAL", False)):
        # Para quem está posicionado: sugestão é vender/realizar
        # Mantemos SL/TP do ponto atual como guia inverso
        return "VENDER/REALIZAR", price, price + atr_mult_sl * atr_val, price - atr_mult_tp * atr_val
    return "ESPERAR", price, sl, tp

# ---------------- Controls ----------------
colA, colB, colC, colD = st.columns([1,1,1,1])
quote = colA.selectbox("Moeda de cotação", ["BRL", "USDT"], index=0)
timeframe = colB.selectbox("Timeframe", ["15m", "5m", "1h"], index=0)
topn = int(colC.number_input("Quantos pares varrer (por volume)", min_value=10, max_value=250, value=60, step=10))
lookback = st.slider("Lookback breakout (barras)", 10, 50, 20, 1)
vol_mult = st.slider("Múltiplo de volume", 1.0, 3.0, 1.6, 0.1)
rsi_thr = st.slider("RSI mínimo", 40, 60, 50, 1)
atr_mult_sl = st.slider("ATR p/ Stop (x)", 0.5, 2.0, 1.0, 0.1)
atr_mult_tp = st.slider("ATR p/ Take (x)", 1.5, 5.0, 3.0, 0.5)

st.write("🔎 Carregando universo e varrendo pares por volume…")
universe = get_top_by_volume(quote=quote, topn=topn)

if not universe:
    st.error("Não consegui carregar os pares. Tente novamente em alguns segundos.")
    st.stop()

st.success(f"Total de pares varridos: {len(universe)} ({quote})")

# Scan
progress = st.progress(0, text="Varrendo OHLCV e calculando sinais…")
rows_buy = []
rows_sell = []

for i, sym in enumerate(universe, start=1):
    try:
        df = load_ohlcv(sym, timeframe=timeframe, limit=300)
        d = compute_signals(df, lookback, vol_mult, rsi_thr)
        last = d.iloc[-1]
        action, price, sl, tp = make_decision(last, atr_mult_sl, atr_mult_tp)

        # Score para ranking (simples e transparente)
        score = 0
        if action == "COMPRAR":
            score += 2
            if last["close"] > last["VWAP"]: score += 1
            if last["RSI"] > (rsi_thr + 5): score += 1
            exp_rr = (tp - price) / max(price - sl, 1e-9)
            rows_buy.append([sym, float(price), float(sl), float(tp), float(last["RSI"]), float(last["ATR"]), float(last["VWAP"]), round(exp_rr,2), score])
        elif action == "VENDER/REALIZAR":
            score += 2
            if last["close"] < last["VWAP"]: score += 1
            if last["RSI"] < 45: score += 1
            rows_sell.append([sym, float(price), float(sl), float(tp), float(last["RSI"]), float(last["ATR"]), float(last["VWAP"]), score])
    except Exception as e:
        # Falha de API ou símbolo sem OHLCV
        pass
    progress.progress(i/len(universe), text=f"Varrendo… {i}/{len(universe)}")

progress.empty()

# Tables
def fmt_money(x):
    return f"{x:,.6f}".replace(",", "X").replace(".", ",").replace("X",".")

if rows_buy:
    df_buy = pd.DataFrame(rows_buy, columns=["Par","Preço","SL","TP","RSI","ATR","VWAP","R:R","Score"]).sort_values(["Score","R:R","RSI"], ascending=[False,False,False]).head(20)
    df_buy["Preço"] = df_buy["Preço"].map(fmt_money); df_buy["SL"] = df_buy["SL"].map(fmt_money); df_buy["TP"] = df_buy["TP"].map(fmt_money); df_buy["VWAP"] = df_buy["VWAP"].map(fmt_money)
    st.subheader("🟢 Top oportunidades de **COMPRA** agora")
    st.dataframe(df_buy, use_container_width=True)
else:
    st.info("Sem sinais fortes de COMPRA neste instante.")

if rows_sell:
    df_sell = pd.DataFrame(rows_sell, columns=["Par","Preço","SL (inverso)","TP (inverso)","RSI","ATR","VWAP","Score"]).sort_values(["Score","RSI"], ascending=[False,True]).head(20)
    df_sell["Preço"] = df_sell["Preço"].map(fmt_money); df_sell["VWAP"] = df_sell["VWAP"].map(fmt_money)
    st.subheader("🔴 Top oportunidades de **VENDA / REALIZAR** agora")
    st.dataframe(df_sell, use_container_width=True)
else:
    st.info("Sem sinais fortes de VENDA neste instante.")

st.caption(f"Atualizado: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC — Fonte: Binance via ccxt")
