import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def compute_technical_analysis(df, ma_s_val=20, ma_l_val=50):
    """Calcule tout le socle technique"""
    df = df.copy()
    df['X'] = np.arange(len(df))

    # Régression Linéaire
    model_lt = LinearRegression().fit(df[['X']].values, df['close'].values)
    df['Reg'] = model_lt.predict(df[['X']].values)
    pente_lt = model_lt.coef_[0]
    std_base = (df['close'] - df['Reg']).std()

    # Pente Court Terme (30j)
    prices_ct = df['close'].values[-30:]
    model_ct = LinearRegression().fit(np.arange(len(prices_ct)).reshape(-1, 1), prices_ct)
    pente_ct = model_ct.coef_[0]

    # Moyennes & RSI
    df['MA_S'] = df['close'].rolling(ma_s_val).mean()
    df['MA_L'] = df['close'].rolling(ma_l_val).mean()
    diff = df['close'].diff()
    gain = (diff.where(diff > 0, 0)).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    return df, pente_lt, pente_ct, std_base


def calculate_cycle(df, ecart_canal):
    """Analyse la périodicité des sorties de canal"""
    c_haut, c_bas = df['Reg'] + ecart_canal, df['Reg'] - ecart_canal
    sorties = df[(df['close'] > c_haut) | (df['close'] < c_bas)].index
    if len(sorties) < 2: return 60, 0

    points_cles = [sorties[0]]
    for i in range(1, len(sorties)):
        if (sorties[i] - sorties[i - 1]).days > 10:
            points_cles.append(sorties[i])

    if len(points_cles) > 1:
        intervalles = [(points_cles[i] - points_cles[i - 1]).days for i in range(1, len(points_cles))]
        return np.mean(intervalles) * 2, len(points_cles)
    return 60, len(points_cles)


def get_final_decision(last_price, last_reg, upper, lower, rsi, pente_lt, pente_ct, proj_reg, ecart):
    """La 'Boîte Noire' de décision"""
    status, marge = "Neutre", 0.0

    if last_price < lower and rsi < 35:
        marge = ((proj_reg + ecart) - last_price) / last_price
        # MARGE_MINIMALE peut être passée en paramètre
        if marge >= 0.05:
            status = "🔥 ACHAT URGENT" if pente_ct > pente_lt * 2 else "🟢 ACHAT"
        else:
            status = "🟠 ATTENDRE"

    elif last_price > upper and rsi > 65:
        marge = (last_price - (proj_reg - ecart)) / last_price
        status = "💥 VENTE URGENTE" if pente_ct < 0 else "🔴 VENTE"

    return status, round(marge, 4)


def prepare_visual_metrics(funds, last_price):
    """Transforme les données fondamentales en infos visuelles (couleurs/labels)"""

    # Logic Valorisation
    target = funds.get('targetPrice', 0)
    val_info = {"target": target, "label": "N/A", "color": "#FFFFFF"}
    if target > 0:
        diff = (target / last_price) - 1
        if diff > 0.05:
            val_info.update({"label": "SOUS-ÉVALUÉ", "color": "#00FF00"})
        elif diff < -0.05:
            val_info.update({"label": "SUR-ÉVALUÉ", "color": "#FF0000"})
        else:
            val_info.update({"label": "ÉQUITABLE", "color": "#FFFF00"})

    # Logic Dette
    debt = funds.get('debtToEquity')
    debt_info = {"ratio": debt, "label": "N/A", "color": "#FFFFFF"}
    if debt is not None:
        if debt < 50:
            debt_info.update({"label": "SOLIDE", "color": "#00FF00"})
        elif debt < 150:
            debt_info.update({"label": "MOYEN", "color": "#FFFF00"})
        else:
            debt_info.update({"label": "RISQUÉ", "color": "#FF0000"})

    return val_info, debt_info
