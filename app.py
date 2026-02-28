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
st.set_page_config(page_title="Analyse Stratégique & Opportunity", layout="wide")


def load_config():
    if os.path.exists("config_tickers.csv"):
        return pd.read_csv("config_tickers.csv", sep=";")
    return pd.DataFrame({'Nom': ['Exemple'], 'Ticker': ['AAPL'], 'Coef': [1.0], 'Days': [730]})


def calculer_cycle_majeur(df, ecart_canal):
    c_haut = df['Reg'] + ecart_canal
    c_bas = df['Reg'] - ecart_canal

    # Identification des sorties réelles
    sorties = df[(df['Close'] > c_haut) | (df['Close'] < c_bas)].index

    if len(sorties) < 2:
        return 60, 0  # 60 jours par défaut, 0 dépassement

    # Filtrage pour ne garder que le début de chaque incursion
    points_cles = [sorties[0]]
    for i in range(1, len(sorties)):
        if (sorties[i] - sorties[i - 1]).days > 10:
            points_cles.append(sorties[i])

    nb_depassements = len(points_cles)

    if nb_depassements > 1:
        intervalles = [(points_cles[i] - points_cles[i - 1]).days for i in range(1, len(points_cles))]
        cycle = np.mean(intervalles) * 2
        return cycle, nb_depassements

    return 60, nb_depassements

# --- INITIALISATION ---
if 'config_df' not in st.session_state:
    st.session_state.config_df = load_config()

# --- SIDEBAR : SÉLECTION & RÉGLAGES ---
st.sidebar.title("🏛️ Menu & Opportunity")
df_conf = st.session_state.config_df

liste_recherche = [f"{row['Nom']} [{row['Ticker']}]" for _, row in df_conf.iterrows()]
selected_label = st.sidebar.selectbox("Sélectionner pour réglage", liste_recherche)

idx = liste_recherche.index(selected_label)
row = df_conf.iloc[idx]

st.sidebar.markdown("---")
st.sidebar.subheader(f"Réglages : {row['Ticker']}")

# L'astuce est ici : on ajoute 'key=row['Ticker']' pour forcer le reset du slider
# à chaque changement de sélection dans la combobox.
new_coef = st.sidebar.slider(
    "Ajuster le Coef du titre",
    0.5, 5.0,
    float(row['Coef']),
    step=0.05,
    key=f"slider_{row['Ticker']}"
)
view_days = st.sidebar.slider(
    "Historique (Jours)", 60, 2000, int(row['Days']), step=10, key=f"d_{row['Ticker']}"
)
ma_s_val = st.sidebar.slider("Moyenne Courte", 5, 100, 20, key=f"ma_s_{row['Ticker']}")
ma_l_val = st.sidebar.slider("Moyenne Longue", 20, 300, 50, key=f"ma_l_{row['Ticker']}")

# --- TRAITEMENT DES DONNÉES ---
ticker = row['Ticker']
df_raw = yf.download(ticker, start=datetime.now() - timedelta(days=view_days), progress=False, auto_adjust=True)

if not df_raw.empty:
    close_vals = df_raw['Close'].iloc[:, 0] if isinstance(df_raw['Close'], pd.DataFrame) else df_raw['Close']
    df = pd.DataFrame({'Close': close_vals})
    df['X'] = np.arange(len(df))

    # 1. Régression LT
    model_lt = LinearRegression().fit(df[['X']].values, df['Close'].values)
    df['Reg'] = model_lt.predict(df[['X']].values)
    pente_lt = model_lt.coef_[0]
    std_base = (df['Close'] - df['Reg']).std()

    # Application du Coef dynamique du slider
    ecart_reel = std_base * new_coef
    df['Upper'] = df['Reg'] + ecart_reel
    df['Lower'] = df['Reg'] - ecart_reel

    # 2. Pente Court Terme (30j)
    prices_ct = df['Close'].values[-30:]
    model_ct = LinearRegression().fit(np.arange(len(prices_ct)).reshape(-1, 1), prices_ct)
    pente_ct = model_ct.coef_[0]

    # 3. Moyennes & RSI
    df['MA_S'] = df['Close'].rolling(ma_s_val).mean()
    df['MA_L'] = df['Close'].rolling(ma_l_val).mean()
    diff = df['Close'].diff()
    gain = (diff.where(diff > 0, 0)).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # 4. Cycle & Marge
    cycle_j, n_sorties = calculer_cycle_majeur(df, ecart_reel)
    proj_reg = df['Reg'].iloc[-1] + (pente_lt * cycle_j)

    status = "Neutre"
    marge_est = 0.0
    last_price = df['Close'].iloc[-1]

    if last_price < df['Lower'].iloc[-1] and df['RSI'].iloc[-1] < 35:
        marge_est = ((proj_reg + ecart_reel) - last_price) / last_price
        status = "🔥 ACHAT URGENT" if pente_ct > pente_lt * 2 else "🟢 ACHAT"
    elif last_price > df['Upper'].iloc[-1] and df['RSI'].iloc[-1] > 65:
        marge_est = (last_price - (proj_reg - ecart_reel)) / last_price
        status = "💥 VENTE URGENTE" if (pente_ct < pente_lt * 2 and pente_ct < 0) else "🔴 VENTE"

    # --- AFFICHAGE ---
    def get_color(s):
        return {"🔥 ACHAT URGENT": "#00FF00", "💥 VENTE URGENTE": "#FF0000", "🟢 ACHAT": "#ADFF2F",
                "🔴 VENTE": "#FF4500"}.get(s, "#FFFFFF")
#______________________________
    # --- INSERTION ICI : ANALYSE FONDAMENTALE ---
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        # 1. VALEUR ESTIMÉE
        valeur_estimee = info.get('targetMeanPrice') or (info.get('bookValue', 0) * 1.5)

        # 2. RATIO DE DETTE (D/E)
        debt_equity = info.get('debtToEquity')

        # Logique de couleur pour la Valorisation
        if valeur_estimee and valeur_estimee > 0:
            if last_price < (valeur_estimee * 0.95):
                color_val, label_val = "#00FF00", "SOUS-ÉVALUÉ"
            elif last_price > (valeur_estimee * 1.05):
                color_val, label_val = "#FF0000", "SUR-ÉVALUÉ"
            else:
                color_val, label_val = "#FFFF00", "PRIX ÉQUITABLE"
        else:
            valeur_estimee, label_val, color_val = 0, "N/A", "#FFFFFF"

        # Logique de couleur pour la Dette
        if debt_equity is not None:
            if debt_equity < 50:
                color_debt, label_debt = "#00FF00", "SOLIDE"
            elif debt_equity < 150:
                color_debt, label_debt = "#FFFF00", "MOYEN"
            else:
                color_debt, label_debt = "#FF0000", "RISQUÉ"
        else:
            label_debt, color_debt = "N/A", "#FFFFFF"
    except:
        valeur_estimee, label_val, color_val = 0, "Erreur", "#FFFFFF"
        debt_equity, label_debt, color_debt = None, "Erreur", "#FFFFFF"

    # --- AFFICHAGE DU STATUT ---
    st.markdown(f"<h1 style='text-align: center; color: {get_color(status)};'>{status}</h1>",
                unsafe_allow_html=True)
    # --- AFFICHAGE SYNTHÉTIQUE (6 COLONNES) ---
    st.markdown("---")

    # Création des 6 colonnes
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    # 1. Le Prix (Analyse Directe)
    c1.metric("Prix Actuel", f"{last_price:.3f}")

    # 2. Valorisation (Fondamental)
    # On utilise la couleur définie dans la logique précédente pour le label
    c2.markdown(f"**Valorisation**")
    c2.markdown(f"<span style='color:{color_val}; font-weight:bold;'>{label_val}</span>", unsafe_allow_html=True)
    c2.write(f"Est: {valeur_estimee:.2f}" if valeur_estimee > 0 else "Est: N/A")

    # 3. Santé Financière (Fondamental)
    c3.markdown(f"**Santé (D/E)**")
    c3.markdown(f"<span style='color:{color_debt}; font-weight:bold;'>{label_debt}</span>", unsafe_allow_html=True)
    c3.write(f"Ratio: {debt_equity:.1f}%" if debt_equity else "Ratio: N/A")

    # 4. Marge Potentielle (Opportunity)
    c4.metric("Marge Canal", f"{marge_est * 100:.1f}%")

    # 5. Cycle (Timing)
    c5.metric("Cycle (Majeur)", f"{int(cycle_j)} j")
    color_fiab = "#00FF00" if n_sorties >= 5 else "#FFFF00" if n_sorties >= 3 else "#FF0000"
    c5.markdown(f"Sorties : <span style='color:{color_fiab}; font-weight:bold;'>{n_sorties}</span>",
                unsafe_allow_html=True)

    # 6. RSI (Momentum)
    rsi_val = int(df['RSI'].iloc[-1])
    # Petit indicateur visuel pour le RSI
    rsi_status = "📉" if rsi_val < 30 else "📈" if rsi_val > 70 else "⚖️"
    c6.metric("RSI (14j)", f"{rsi_val}", rsi_status)

    st.markdown("---")

    # Graphique Triple Étage
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.3, 0.2])

    # 1. Régression
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Prix", line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reg'], name="Tendance", line=dict(color='orange', dash='dot')), row=1,
                  col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Upper'], name="Canal +", line=dict(color='rgba(255,0,0,0.3)', width=0.5)), row=1,
        col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name="Canal -", fill='tonexty', fillcolor='rgba(0,255,0,0.05)',
                             line=dict(color='rgba(0,255,0,0.3)', width=0.5)), row=1, col=1)

    # 2. Moyennes
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_S'], name="MA Courte", line=dict(color='cyan')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_L'], name="MA Longue", line=dict(color='magenta')), row=2, col=1)

    # 3. RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='yellow')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=850, template="plotly_dark")
    st.plotly_chart(fig, width='stretch')

    st.sidebar.markdown(f"**Pente LT :** `{pente_lt:.6f}`")
    st.sidebar.markdown(f"**Pente 30j :** `{pente_ct:.6f}`")

else:
    st.error("Données Yahoo Finance indisponibles.")
