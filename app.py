import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Analyse Stratégique & Opportunity", layout="wide")


def load_config():
    if os.path.exists("config_tickers.csv"):
        return pd.read_csv("config_tickers.csv", sep=";")
    return pd.DataFrame({'Nom': ['Air Liquide'], 'Ticker': ['AI.PA'], 'Coef': [1.0], 'Jours': [730]})


if 'config_df' not in st.session_state:
    st.session_state.config_df = load_config()

# --- SIDEBAR & COMBOBOX ---
st.sidebar.title("🏛️ Menu & Opportunity")
df_conf = st.session_state.config_df

# Combobox : Nom [Ticker]
liste_recherche = [f"{row['Nom']} [{row['Ticker']}]" for _, row in df_conf.iterrows()]
selected_label = st.sidebar.selectbox("Rechercher une valeur", liste_recherche)

# Extraction de la ligne sélectionnée
selected_nom = selected_label.split(" [")[0]
row = df_conf[df_conf['Nom'] == selected_nom].iloc[0]

# Sliders de paramètres
vol_mult = st.sidebar.slider("Écart Canal (Coef)", 0.5, 4.0, float(row['Coef']))
days_hist = st.sidebar.slider("Historique (jours)", 60, 1500, int(row['Days']))
ma_short = st.sidebar.slider("Moyenne Courte", 5, 100, 20)
ma_long = st.sidebar.slider("Moyenne Longue", 20, 300, 50)

# --- CHARGEMENT DONNÉES ---
ticker = row['Ticker']
df_raw = yf.download(ticker, start=datetime.now() - timedelta(days=days_hist), progress=False, auto_adjust=True)

if not df_raw.empty:
    # Nettoyage yfinance
    close_vals = df_raw['Close'].iloc[:, 0] if isinstance(df_raw['Close'], pd.DataFrame) else df_raw['Close']
    df = pd.DataFrame({'Close': close_vals})
    prices = df['Close'].values

    # 1. RÉGRESSION LONG TERME (LT)
    df['X'] = np.arange(len(df))
    model_lt = LinearRegression().fit(df[['X']].values, prices)
    df['Reg'] = model_lt.predict(df[['X']].values)
    pente_lt = model_lt.coef_[0]
    std = (df['Close'] - df['Reg']).std()
    df['Upper'] = df['Reg'] + (std * vol_mult)
    df['Lower'] = df['Reg'] - (std * vol_mult)

    # 2. RÉGRESSION COURT TERME (30j)
    prices_ct = prices[-30:] if len(prices) >= 30 else prices
    x_ct = np.arange(len(prices_ct)).reshape(-1, 1)
    model_ct = LinearRegression().fit(x_ct, prices_ct)
    pente_ct = model_ct.coef_[0]

    # 3. MOYENNES MOBILES & RSI
    df['MA_S'] = df['Close'].rolling(ma_short).mean()
    df['MA_L'] = df['Close'].rolling(ma_long).mean()

    diff = df['Close'].diff()
    gain = (diff.where(diff > 0, 0)).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # 4. CYCLE (find_peaks)
    peaks, _ = find_peaks(prices, distance=20)
    lows, _ = find_peaks(-prices, distance=20)
    extrema = np.sort(np.concatenate([peaks, lows]))
    cycle_moyen = np.mean(np.diff(extrema)) if len(extrema) > 1 else 40

    # 5. LOGIQUE OPPORTUNITY
    last_price = prices[-1]
    last_reg = df['Reg'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    projection = last_reg + (pente_lt * cycle_moyen)
    marge_min = 0.05
    status = "Neutre"
    marge_estimee = 0.0

    if last_price < (last_reg - std * vol_mult) and last_rsi < 35:
        marge_estimee = ((projection + std) - last_price) / last_price
        if marge_estimee >= marge_min:
            status = "🔥 ACHAT URGENT" if pente_ct > pente_lt * 2 else "🟢 ACHAT"
        else:
            status = "🟠 ATTENDRE"
    elif last_price > (last_reg + std * vol_mult) and last_rsi > 65:
        marge_estimee = (last_price - (projection - std)) / last_price
        if marge_estimee >= marge_min:
            status = "💥 VENTE URGENTE" if (pente_ct < pente_lt * 2 and pente_ct < 0) else "🔴 VENTE"
        else:
            status = "🔵 CONSERVER"


    # --- AFFICHAGE DES INDICATEURS (METRICS) ---
    def get_color(s):
        return {"🔥 ACHAT URGENT": "#00FF00", "💥 VENTE URGENTE": "#FF0000", "🟢 ACHAT": "#ADFF2F",
                "🔴 VENTE": "#FF4500"}.get(s, "#FFFFFF")


    st.markdown(f"<h1 style='text-align: center; color: {get_color(status)};'>{status}</h1>", unsafe_allow_html=True)

    # Ligne de Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prix Actuel", f"{last_price:.3f}", f"{((last_price / df['Close'].iloc[-2]) - 1) * 100:.2f}%")
    m2.metric("RSI (14j)", f"{int(last_rsi)}",
              "Suracheté" if last_rsi > 70 else "Survendu" if last_rsi < 30 else "Neutre")
    m3.metric("Marge Est.", f"{marge_estimee * 100:.1f}%")
    m4.metric("Cycle Moyen", f"{int(cycle_moyen)} j")

    # --- GRAPHIQUE ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.3, 0.2])

    # Canal Régression
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Prix", line=dict(color='white', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reg'], name="Tendance", line=dict(color='orange', dash='dot')), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name="Canal +", line=dict(color='rgba(255,0,0,0.3)', width=0)),
                  row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Lower'], name="Canal -", fill='tonexty', fillcolor='rgba(100,255,100,0.1)',
                   line=dict(color='rgba(0,255,0,0.3)', width=0)), row=1, col=1)

    # Moyennes Mobiles
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_S'], name=f"MA {ma_short}", line=dict(color='cyan')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_L'], name=f"MA {ma_long}", line=dict(color='magenta')), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='yellow')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=850, template="plotly_dark", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width='stretch')

    # Sidebar Info Pente
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Pente 730j :** `{pente_lt:.6f}`")
    st.sidebar.write(f"**Pente 30j :** `{pente_ct:.6f}`")

else:
    st.error("Données indisponibles pour ce ticker.")
