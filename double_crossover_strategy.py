import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import quantstats_lumi as qs
from tqdm import tqdm

from datetime import datetime

from pandas import Series, DataFrame

from utils import *
#plt.style.use("fivethirtyeight")
#plt.style.use("ggplot")
plt.style.use("classic")
figsize = (14, 6)

class DoubleCrossoverStrategy:
    def __init__(self,
                 stock: str,
                 start_date: str | pd.Timestamp,
                 end_date: str | pd.Timestamp = pd.Timestamp(datetime.today()),
                 short_ma: int = 20,
                 long_ma: int = 50):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.strategy_df = self.compute_strategy()

    def compute_sma_df(self, drop_na: bool = True) -> pd.DataFrame:
        return compute_sma_df(self.stock, self.start_date, self.end_date, self.short_ma, self.long_ma, drop_na)

    def compute_buy_sell_series(self) -> pd.Series:
        df = self.compute_sma_df()
        result_day = [df.index[0]]
        result_action = ["wait"]
        for i in range(1, len(df)):
            if df.iloc[i - 1, 1] < df.iloc[i - 1, 2] and df.iloc[i, 1] >= df.iloc[i, 2]:
                result_day.append(df.index[i])
                result_action.append("buy")
            elif df.iloc[i - 1, 1] > df.iloc[i - 1, 2] and df.iloc[i, 1] <= df.iloc[i, 2]:
                result_day.append(df.index[i])
                result_action.append("sell")
            else:
                result_action.append("wait")
                result_day.append(df.index[i])
        return pd.Series(result_action, index=result_day)

    def compute_strategy(self, starting_capital: float = 1.0) -> pd.DataFrame:
        df = self.compute_sma_df()
        df["Action"] = self.compute_buy_sell_series()
        df[f"{self.stock}_Returns"] = compute_returns(df[self.stock])

        portfolio = [starting_capital]
        active: bool = False
        for i in range(1, len(df)):

            if not active:
                portfolio.append(portfolio[-1])
            else:
                portfolio.append(portfolio[-1] * (1 + df[f"{self.stock}_Returns"].iloc[i]))

            if df["Action"].iloc[i] == "buy":
                active = True
            if df["Action"].iloc[i] == "sell":
                active = False

        df["Strategy"] = portfolio
        df["Strategy_Returns"] = compute_returns(df["Strategy"])
        return df

    def optimize_parameters(self, max_short: int = 50, max_long: int = 200, update_self: bool = False):
        short_ma = self.short_ma
        long_ma = self.long_ma
        avg_strategy_return = qs.stats.avg_return(self.strategy_df["Strategy_Returns"])
        for i in tqdm(range(10, max_short + 1)):
            for j in range(i + 10, max_long + 1):
                x: float = qs.stats.avg_return(DoubleCrossoverStrategy(self.stock,
                                                                       self.start_date,
                                                                       self.end_date,
                                                                       i,
                                                                       j).strategy_df["Strategy_Returns"])
                if x >= avg_strategy_return:
                    avg_strategy_return = x
                    short_ma = i
                    long_ma = j
        print(f"Optimal parameters:\nShort ma: {short_ma}\nLong ma: {long_ma}")
        if update_self:
            self.short_ma = short_ma
            self.long_ma = long_ma
            self.strategy_df = self.compute_strategy()
        return short_ma, long_ma

    def plot_ma(self, ax):

        self.strategy_df.iloc[:, 0:3].plot(ax=ax)

        for day in self.strategy_df.index:
            action = self.strategy_df["Action"].loc[day]

            if action == "sell":
                ax.axvline(day, color="red", linestyle="--", label="sell")

            elif action == "buy":
                ax.axvline(day, color="blue", linestyle="--", label="buy")

    def plot_strategy(self, ax, strategy_color: str = "red", stock_color: str = "blue"):
        stock = self.strategy_df[self.stock] / self.strategy_df[self.stock].iloc[0]
        strategy = self.strategy_df["Strategy"] / self.strategy_df["Strategy"].iloc[0]
        ax.plot(stock, color=stock_color, label=self.stock)
        ax.plot(strategy, color=strategy_color, label="Strategy")

