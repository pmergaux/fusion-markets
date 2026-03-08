import streamlit as st
import os
import pandas as pd

# Import des modules
from engine_data import get_clean_data, get_robust_fundamentals
from engine_strategy import (compute_technical_analysis, calculate_cycle,
                             get_final_decision, prepare_visual_metrics)
from ui_components import (display_sidebar_controls, display_status_header,
                           display_metrics_bar, display_main_chart, display_technical_info)

st.set_page_config(page_title="Analyse Stratégique Pro", layout="wide")

# 1. Chargement Configuration
if 'config_df' not in st.session_state:
    if os.path.exists("config_tickers.csv"):
        st.session_state.config_df = pd.read_csv("config_tickers.csv", sep=";")
    else:
        st.error("Fichier config_tickers.csv manquant.")
        st.stop()

# 2. Sidebar Modulaire (Récupère ticker, coef, days, ma_s, ma_l)
params = display_sidebar_controls(st.session_state.config_df)
df_config = st.session_state.config_df
ticker_sel = params['ticker']
# 3. Identification de la ligne
indices = df_config.index[df_config['Ticker'] == ticker_sel].tolist()
if not indices or len(indices) == 0:
    st.error(f"Le ticker {ticker_sel} n'est pas dans le CSV.")
    st.stop()
# 3. Récupération des données
df, t_obj = get_clean_data(params['ticker'], params['days'])
idx_ligne = indices[0]

if df is not None:
    # 4. Calculs Stratégiques
    df, p_lt, p_ct, std = compute_technical_analysis(df, params['ma_s'], params['ma_l'])

    ecart = std * params['coef']
    df['Upper'], df['Lower'] = df['Reg'] + ecart, df['Reg'] - ecart

    cycle_j, n_sorties = calculate_cycle(df, ecart)
    funds = get_robust_fundamentals(ticker_sel, df_config.iloc[idx_ligne])

    # 5. Calcul des indicateurs visuels (Dette et Valo)
    last_price = df['close'].iloc[-1]
    val_info, debt_info = prepare_visual_metrics(funds, last_price)

    # 6. Décision Finale
    proj_reg = df['Reg'].iloc[-1] + (p_lt * cycle_j)
    status, marge_val = get_final_decision(
        last_price, df['Reg'].iloc[-1], df['Upper'].iloc[-1], df['Lower'].iloc[-1],
        df['RSI'].iloc[-1], p_lt, p_ct, proj_reg, ecart
    )

    # 7. Affichage UI
    display_status_header(status)

    cycle_data = {"days": cycle_j, "sorties": n_sorties}
    display_metrics_bar(last_price, val_info, debt_info, marge_val, cycle_data, df['RSI'].iloc[-1])

    display_main_chart(df)
    display_technical_info(p_lt, p_ct)

else:
    st.error(f"Erreur de chargement pour {params['ticker']}")
