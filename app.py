import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Comparateur Stratégique", layout="wide")


def load_config():
    if os.path.exists("config_tickers.csv"):
        return pd.read_csv("config_tickers.csv", sep=";")
    return pd.DataFrame({'Nom': ['Air Liquide'], 'Ticker': ['AI.PA'], 'Coef': [1.0], 'Days': [730]})


if 'config_df' not in st.session_state:
    st.session_state.config_df = load_config()

# --- SIDEBAR ---
st.sidebar.title("🏛️ Menu & Analyse")
df_conf = st.session_state.config_df
selected_nom = st.sidebar.selectbox("Valeur à analyser", df_conf['Nom'].tolist())
row = df_conf[df_conf['Nom'] == selected_nom].iloc[0]

# Paramètres dynamiques du CSV
vol_mult = st.sidebar.slider("Écart Canal (Coef)", 0.5, 4.0, float(row['Coef']))
days_hist = st.sidebar.slider("Historique (jours)", 60, 1500, int(row['Days']) if pd.notnull(row['Days']) else 730)
ma_short = st.sidebar.slider("Moyenne Courte", 5, 100, 20)
ma_long = st.sidebar.slider("Moyenne Longue", 20, 300, 50)

# --- CHARGEMENT DONNÉES ---
ticker = row['Ticker']
df_raw = yf.download(ticker, start=datetime.now() - timedelta(days=days_hist), progress=False)

if not df_raw.empty:
    df = pd.DataFrame(
        {'Close': df_raw['Close'].iloc[:, 0] if isinstance(df_raw['Close'], pd.DataFrame) else df_raw['Close']})

    # 1. Calcul Régression (Canal)
    df['X'] = np.arange(len(df))
    model = LinearRegression().fit(df[['X']].values, df['Close'].values)
    df['Reg'] = model.predict(df[['X']].values)
    std = (df['Close'] - df['Reg']).std()
    df['Upper'] = df['Reg'] + (std * vol_mult)
    df['Lower'] = df['Reg'] - (std * vol_mult)

    # 2. Calcul Moyennes
    df['MA_S'] = df['Close'].rolling(ma_short).mean()
    df['MA_L'] = df['Close'].rolling(ma_long).mean()

    # 3. Calcul RSI
    diff = df['Close'].diff()
    gain = (diff.where(diff > 0, 0)).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # --- GRAPHIQUE TRIPLE ÉTAGE ---
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(f"1. RÉGRESSION (Canal Coef {vol_mult})", "2. MOYENNES MOBILES (Croisement)", "3. RSI (Force)"),
        row_heights=[0.4, 0.4, 0.2]
    )

    # Étage 1 : Régression
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Prix", line=dict(color='white', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reg'], name="Tendance", line=dict(color='orange', dash='dot')), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name="Canal +", line=dict(color='red', width=1, dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="Canal -", line=dict(color='green', width=1, dash='dash')),
                  row=1, col=1)

    # Étage 2 : Moyennes
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], showlegend=False, line=dict(color='white', width=1)), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_S'], name=f"MA {ma_short}", line=dict(color='cyan')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_L'], name=f"MA {ma_long}", line=dict(color='magenta')), row=2, col=1)

    # Étage 3 : RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='yellow')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=900, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

    # Comparaison de performance rapide
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"💡 Comparaison : La régression montre une tendance {'HAUSSIÈRE' if df['Reg'].iloc[-1] > df['Reg'].iloc[0] else 'BAISSIÈRE'}.")

else:
    st.error("Données Yahoo Finance indisponibles.")

# Lien vers le site PHP
st.sidebar.markdown(f"🔗 [Accéder au Manifeste](https://transparence-simplicite.ovh)")
