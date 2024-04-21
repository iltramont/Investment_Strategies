import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime


def compute_sma_df(stock: str,
                   start: str | pd.Timestamp,
                   end: str | pd.Timestamp = pd.Timestamp(datetime.today()),
                   short_ma: int = 20,
                   long_ma: int = 50,
                   drop_na: bool = False) -> pd.DataFrame:
    stock_close: pd.Series = yf.Ticker(stock).history(start=start, end=end)["Close"]
    df: pd.DataFrame = pd.DataFrame({
        stock: stock_close,
        f"ma_{short_ma}": stock_close.rolling(short_ma).mean(),
        f"ma_{long_ma}": stock_close.rolling(long_ma).mean()
    })
    df.index = pd.to_datetime(df.index.strftime('%Y-%m-%d'))
    if drop_na:
        return df.dropna()
    else:
        return df


def compute_returns(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return (df - df.shift(1)) / df.shift(1)


def compute_yearly_compounded_returns(series: pd.Series, prices: bool) -> float:
    years = (series.index[-1] - series.index[0]).days / 365

    if prices:
        total_return = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
    else:
        total_return = (series + 1).prod() - 1

    return (1 + total_return) ** (1 / years) - 1


def get_ew_number_of_shares_to_buy(date: str | pd.Timestamp,
                                   tickers: list[str],
                                   capital: float,
                                   tickers_data: pd.DataFrame):
    return ((capital / float(len(tickers))) / tickers_data['Close'][tickers].loc[date:].iloc[0]) // 1
