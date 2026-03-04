import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd

def load_config_session(st):
    if 'config_df' not in st.session_state:
        if os.path.exists("config_tickers.csv"):
            st.session_state.config_df = pd.read_csv("config_tickers.csv", sep=";")
        else:
            st.error("Fichier config_tickers.csv manquant.")
            st.stop()
            return False
    return True

def display_sidebar_controls(config_df, title="🏛️ Menu & Stratégie"):
    """Gère toute la navigation et les réglages dans la sidebar"""
    st.sidebar.title(title)

    # 1. Sélection de l'actionst.sidebar.title()
    labels = [f"{row['Nom']} [{row['Ticker']}]" for _, row in config_df.iterrows()]
    selected_label = st.sidebar.selectbox("Action à analyser", labels)

    # Trouver l'index pour récupérer les valeurs par défaut du CSV
    idx = labels.index(selected_label)
    row_cfg = config_df.iloc[idx]
    ticker_sym = row_cfg['Ticker']

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Réglages : {ticker_sym}")

    # 2. Sliders de réglages (on utilise les clés dynamiques pour éviter les conflits)
    new_coef = st.sidebar.slider("Coefficient Canal", 0.5, 5.0, float(row_cfg['Coef']), 0.05, key=f"s_{ticker_sym}")
    view_days = st.sidebar.slider("Historique (Jours)", 60, 2000, int(row_cfg['Days']), 10, key=f"d_{ticker_sym}")
    ma_s = st.sidebar.slider("Moyenne Courte", 5, 100, 20)
    ma_l = st.sidebar.slider("Moyenne Longue", 20, 300, 50)

    # On retourne un dictionnaire de paramètres
    return {
        "ticker": ticker_sym,
        "coef": new_coef,
        "days": view_days,
        "ma_s": ma_s,
        "ma_l": ma_l
    }


def display_technical_info(p_lt, p_ct):
    """Petit widget en bas de sidebar pour les pentes"""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Pente Long Terme :** `{p_lt:.6f}`")
    st.sidebar.markdown(f"**Pente 30j :** `{p_ct:.6f}`")

def display_status_header(status):
    """Affiche le bandeau de statut coloré en haut de page"""
    colors = {
        "🔥 ACHAT URGENT": "#00FF00", "💥 VENTE URGENTE": "#FF0000",
        "🟢 ACHAT": "#ADFF2F", "🔴 VENTE": "#FF4500", "Neutre": "#FFFFFF"
    }
    color = colors.get(status, "#FFFFFF")
    st.markdown(f"<h1 style='text-align: center; color: {color};'>{status}</h1>", unsafe_allow_html=True)


def display_metrics_bar(last_price, val_info, debt_info, marge, cycle_info, rsi_val):
    """Affiche les 6 colonnes de metrics standardisées"""
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Prix Actuel", f"{last_price:.2f}")

    # Valorisation
    c2.markdown("**Valorisation**")
    c2.markdown(f"<span style='color:{val_info['color']}; font-weight:bold;'>{val_info['label']}</span>",
                unsafe_allow_html=True)
    c2.write(f"Cible: {val_info['target']:.2f}" if val_info['target'] > 0 else "Cible: N/A")

    # Santé
    c3.markdown("**Santé (D/E)**")
    c3.markdown(f"<span style='color:{debt_info['color']}; font-weight:bold;'>{debt_info['label']}</span>",
                unsafe_allow_html=True)
    c3.write(f"Ratio: {debt_info['ratio']:.1f}%" if debt_info['ratio'] is not None else "Ratio: N/A")

    c4.metric("Marge Canal", f"{marge * 100:.1f}%")
    c5.metric("Cycle Estimé", f"{int(cycle_info['days'])} j", f"Sorties: {cycle_info['sorties']}")

    rsi_emoji = "📉" if rsi_val < 30 else "📈" if rsi_val > 70 else "⚖️"
    c6.metric("RSI (14j)", f"{int(rsi_val)}", rsi_emoji)


def display_main_chart(df):
    """Génère le graphique triple étage standardisé"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])

    # Row 1: Prix & Canal
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name="Prix", line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reg'], name="Tendance", line=dict(color='orange', dash='dot')), row=1,
                  col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Upper'], name="Canal +", line=dict(color='rgba(255,0,0,0.3)', width=0.5)), row=1,
        col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="Canal -", fill='tonexty', fillcolor='rgba(0,255,0,0.05)',
                             line=dict(color='rgba(0,255,0,0.3)', width=0.5)), row=1, col=1)

    # Row 2: Moyennes
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_S'], name="MA Courte", line=dict(color='cyan')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_L'], name="MA Longue", line=dict(color='magenta')), row=2, col=1)

    # Row 3: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='yellow')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=800, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')