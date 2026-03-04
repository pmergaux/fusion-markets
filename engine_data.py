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
    """ Centralise la recherche de la dette et de la valorisation """
    fin_data = t_obj.financial_data
    if isinstance(fin_data, dict) and ticker_sym in fin_data:
        data = fin_data[ticker_sym]
        if not isinstance(data, dict): data = {}
    else:
        data = {}

    target = data.get('targetMeanPrice')
    if not target:
        trend = t_obj.recommendation_trend
        if isinstance(trend, dict) and ticker_sym in trend:
            target = trend[ticker_sym].get('targetMeanPrice')

    return {
        "debtToEquity": data.get('debtToEquity'),
        "targetPrice": float(target) if target else 0
    }