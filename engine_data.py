import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from yahooquery import Ticker


def get_clean_data(ticker_sym, days):
    """ Récupère et nettoie les données de YahooQuery """
    t = Ticker(ticker_sym)
    df_raw = t.history(period=f"{days}d", interval="1d")
    if df_raw.empty: return None, None

    df = df_raw.loc[ticker_sym].copy() if isinstance(df_raw.index, pd.MultiIndex) else df_raw.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df, t


def compute_indicators(df, ma_s=20, ma_l=50):
    """ Calcule tous les indicateurs techniques sur le DataFrame """
    df = df.copy()
    df['X'] = np.arange(len(df))
    # Régression
    model = LinearRegression().fit(df[['X']].values, df['close'].values)
    df['Reg'] = model.predict(df[['X']].values)
    # Moyennes et RSI
    df['MA_S'] = df['close'].rolling(ma_s).mean()
    df['MA_L'] = df['close'].rolling(ma_l).mean()
    diff = df['close'].diff()
    gain = (diff.where(diff > 0, 0)).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    return df, model.coef_[0], (df['close'] - df['Reg']).std()


def get_fundamentals(t_obj, ticker_sym):
    # On force la récupération des modules financiers
    all_modules = t_obj.get_modules('financialData recommendationTrend')

    # On initialise avec des valeurs par défaut
    funds_res = {"debtToEquity": None, "targetPrice": 0}

    # 1. Vérification : est-ce qu'on a reçu un dictionnaire pour ce ticker ?
    if isinstance(all_modules, dict) and ticker_sym in all_modules:
        data = all_modules[ticker_sym]

        # 2. Sécurité CRUCIALE : on vérifie que 'data' est bien un dictionnaire
        # Si c'est un 'str', l'erreur AttributeError sera évitée ici
        if isinstance(data, dict):
            # Extraction sécurisée de la dette
            funds_res["debtToEquity"] = data.get('debtToEquity')

            # Extraction du prix cible
            target = data.get('targetMeanPrice')

            # Plan B si target est vide : chercher dans recommendationTrend
            if not target:
                trend = data.get('recommendationTrend')
                if isinstance(trend, dict):
                    target = trend.get('targetMeanPrice', 0)

            # Conversion finale en float pour éviter les surprises
            try:
                funds_res["targetPrice"] = float(target) if target else 0
            except (ValueError, TypeError):
                funds_res["targetPrice"] = 0

    return funds_res


import yfinance as yf

def get_robust_fundamentals(ticker_sym, csv_row):
    """
    Tente de récupérer les infos par priorité décroissante :
    1. YahooQuery (Source principale rapide)
    2. yfinance (Source de secours)
    3. Fichier CSV (Roue de secours si internet renvoie N/A)
    """
    final_target = 0
    final_debt = None

    # --- SOURCE 1 : YAHOOQUERY ---
    try:
        t_yq = Ticker(ticker_sym)
        modules = t_yq.get_modules('financialData')
        if isinstance(modules, dict) and ticker_sym in modules:
            data = modules[ticker_sym]
            if isinstance(data, dict):
                final_target = data.get('targetMeanPrice', 0)
                final_debt = data.get('debtToEquity')
    except:
        pass

    # --- SOURCE 2 : YFINANCE (si toujours vide) ---
    if not final_target or final_debt is None:
        try:
            t_yf = yf.Ticker(ticker_sym)
            info = t_yf.info
            if not final_target:
                final_target = info.get('targetMeanPrice', 0) or info.get('targetLowPrice', 0)
            if final_debt is None:
                final_debt = info.get('debtToEquity')
        except:
            pass

    # --- SOURCE 3 : FICHIER CSV (Dernier recours) ---
    # Si après les deux API on n'a toujours rien, on prend ce qu'il y a dans le CSV
    if not final_target:
        final_target = csv_row.get('Fair_Value', 0)

    if final_debt is None:
        final_debt = csv_row.get('DE_Ratio', None)

    return {
        "targetPrice": float(final_target) if final_target else 0,
        "debtToEquity": float(final_debt) if (final_debt and final_debt > 0) else None
    }
