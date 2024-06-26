import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import utils
import cvxpy as cv
from tqdm import tqdm

from scipy import stats


class QuantitativeMomentum:
    def __init__(self):
        self.history = dict()

    @staticmethod
    def compute_momentum_df(stocks_data: pd.DataFrame,
                            date: str | pd.Timestamp,
                            periods: tuple[int] = (252, 126, 63, 21)) -> pd.DataFrame:

        def _compute_period_return(prices):
            initial_price = prices.iloc[0]
            final_price = prices.iloc[-1]
            return (final_price - initial_price) / initial_price

        # Create DataFrame
        total_tickers = list(stocks_data['Close'].columns)
        df = pd.DataFrame(index=total_tickers, columns=['momentum score'])

        # Compute returns for each period
        for period in periods:
            df[f'{period} day return'] = _compute_period_return(stocks_data['Adj Close'].loc[:date].iloc[-period:])

        # Compute momentum score
        for ticker in df.index:
            scores = []
            for period in periods:
                col_name = f'{period} day return'
                scores.append(stats.percentileofscore(df[col_name], df[col_name].loc[ticker], nan_policy='omit'))
            df.loc[ticker, 'momentum score'] = np.mean(scores)
        return df.sort_values(by='momentum score', ascending=False)

    @staticmethod
    def select_stocks(stocks_data: pd.DataFrame,
                      date: str | pd.Timestamp,
                      n_stocks: int,
                      periods: tuple[int] = (252, 126, 63, 21)) -> list[str]:
        return list(QuantitativeMomentum.compute_momentum_df(stocks_data, date, periods).iloc[:n_stocks].index)

    def strategy_simulation(self, stocks_data: pd.DataFrame,
                            starting_day: str | pd.Timestamp,
                            n_stocks: int,
                            starting_capital: float,
                            rebalance_frequency: int = 30,
                            periods: tuple[int] = (252, 126, 63, 21)) -> pd.Series:

        selected_stocks_dictionary = dict()

        # Select all days
        days = stocks_data.loc[starting_day:].index
        current_day = days[0]

        # Initialize dataframe
        portfolio_cum_ret = pd.DataFrame(index=days, columns=stocks_data['Close'].columns)

        # Initialize series to keep track of available cash
        available_money = pd.Series(index=days)

        # Select stocks
        active_stocks = QuantitativeMomentum.select_stocks(stocks_data, current_day, n_stocks, periods)

        # Add first row
        row = (stocks_data["Close"].loc[current_day:].iloc[0] *
               utils.get_ew_number_of_shares_to_buy(current_day, active_stocks, starting_capital, stocks_data))
        portfolio_cum_ret.loc[current_day, :] = row

        # Compute returns and set parameters for the loop
        stocks_adj_returns = utils.compute_returns(stocks_data['Adj Close'])
        selected_stocks_dictionary[current_day] = active_stocks
        last_rebalance_day = current_day

        # Update available money
        available_money.loc[current_day] = starting_capital - row.sum()

        for day in tqdm(days[1:]):
            # Update cum returns
            stocks_returns = stocks_adj_returns.loc[day]
            portfolio_cum_ret.loc[day, :] = portfolio_cum_ret.loc[current_day, :] * (stocks_returns + 1)

            # Handle period change
            if (day - last_rebalance_day).days > rebalance_frequency:
                last_rebalance_day = day
                # Sell all stocks
                available_money.loc[day] = available_money.loc[current_day] + portfolio_cum_ret.loc[day].sum()
                # Select new stocks
                active_stocks = QuantitativeMomentum.select_stocks(stocks_data, day, n_stocks, periods)
                selected_stocks_dictionary[day] = active_stocks
                # Buy with equal weight
                row = (stocks_data["Close"].loc[day:].iloc[0] *
                       utils.get_ew_number_of_shares_to_buy(day, active_stocks, available_money.loc[day], stocks_data))
                portfolio_cum_ret.loc[day, :] = row
                available_money.loc[day] = available_money.loc[day] - (row.sum())
            else:
                available_money.loc[day] = available_money.loc[current_day]

            # Change current day
            current_day = day

        self.history['available money'] = available_money
        self.history['portfolio cumulative returns'] = portfolio_cum_ret
        self.history['selected stocks'] = pd.DataFrame(selected_stocks_dictionary).T
        self.history['strategy cumulative returns'] = portfolio_cum_ret.sum(axis=1)
        return portfolio_cum_ret.sum(axis=1)


class EasyRebalance:
    def __init__(self):
        self.history = dict()

    def strategy_simulation(self, stocks_data: pd.DataFrame,
                            stocks: list[str],
                            starting_day: str | pd.Timestamp,
                            starting_capital: float,
                            rebalance_frequency: int = 30) -> pd.Series:
        # Select all days
        days = stocks_data.loc[starting_day:].index
        current_day = days[0]

        # Initialize dataframe
        portfolio_cum_ret = pd.DataFrame(index=days, columns=stocks_data['Close'].columns)

        # Initialize series to keep track of available cash
        available_money = pd.Series(index=days)

        # Add first row
        row = (stocks_data["Close"].loc[current_day:].iloc[0] *
               utils.get_ew_number_of_shares_to_buy(current_day, stocks, starting_capital, stocks_data))
        portfolio_cum_ret.loc[current_day, :] = row

        # Compute returns and set parameters for the loop
        stocks_adj_returns = utils.compute_returns(stocks_data['Adj Close'])
        last_rebalance_day = current_day

        # Update available money
        available_money.loc[current_day] = starting_capital - row.sum()

        for day in tqdm(days[1:]):
            # Compute cum returns
            stocks_returns = stocks_adj_returns.loc[day]
            portfolio_cum_ret.loc[day, :] = portfolio_cum_ret.loc[current_day, :] * (stocks_returns + 1)

            # Handle period change
            if (day - last_rebalance_day).days > rebalance_frequency:
                last_rebalance_day = day
                # Sell all stocks
                available_money.loc[day] = available_money.loc[current_day] + portfolio_cum_ret.loc[day].sum()
                # Buy with equal weight
                row = (stocks_data["Close"].loc[day:].iloc[0] *
                       utils.get_ew_number_of_shares_to_buy(day, stocks, available_money.loc[day], stocks_data))
                portfolio_cum_ret.loc[day, :] = row
                available_money[day] = available_money[day] - (row.sum())
            else:
                available_money.loc[day] = available_money.loc[current_day]

            # Change current day
            current_day = day

        self.history['available money'] = available_money
        self.history['portfolio cumulative returns'] = portfolio_cum_ret
        self.history['strategy cumulative returns'] = portfolio_cum_ret.sum(axis=1)
        return portfolio_cum_ret.sum(axis=1)


class InverseVolatility:
    def __init__(self):
        self.history = dict()

    @staticmethod
    def compute_weights(volatility: pd.Series) -> pd.Series:
        return (1 / volatility) / ((1 / volatility).sum())

    def strategy_simulation(self, stocks_data: pd.DataFrame,
                            starting_day: str | pd.Timestamp,
                            starting_capital: float,
                            volatility_window: int = 504,
                            rebalance_frequency: int = 30) -> pd.Series:

        weights_dictionary = dict()

        # Select all days
        days = stocks_data.loc[starting_day:].index
        current_day = days[0]

        # Initialize dataframe
        portfolio_cum_ret = pd.DataFrame(index=days, columns=stocks_data['Close'].columns)

        # Initialize series to keep track of available cash
        available_money = pd.Series(index=days)

        # Compute returns and set parameters for the loop
        stocks_adj_returns = utils.compute_returns(stocks_data['Adj Close'])
        last_rebalance_day = current_day

        # Compute volatility and weights
        volatility = stocks_adj_returns.loc[: current_day].iloc[-volatility_window:].std()
        weights = self.compute_weights(volatility)
        weights_dictionary[current_day] = weights

        # Add first row
        prices = stocks_data["Close"].loc[current_day]
        row = prices * utils.get_weighted_number_of_shares_to_buy(prices, weights, starting_capital)
        portfolio_cum_ret.loc[current_day, :] = row

        # Update available money
        available_money.loc[current_day] = starting_capital - row.sum()

        for day in tqdm(days[1:]):
            # Update cum returns
            stocks_returns = stocks_adj_returns.loc[day]
            portfolio_cum_ret.loc[day, :] = portfolio_cum_ret.loc[current_day, :] * (stocks_returns + 1)

            # Handle period change
            if (day - last_rebalance_day).days > rebalance_frequency:
                last_rebalance_day = day

                # Sell all stocks
                available_money.loc[day] = available_money.loc[current_day] + portfolio_cum_ret.loc[day].sum()

                # Compute volatility and weights
                volatility = stocks_adj_returns.loc[: current_day].iloc[-volatility_window:].std()
                weights = self.compute_weights(volatility)
                weights_dictionary[current_day] = weights

                # Buy stocks
                prices = stocks_data["Close"].loc[day]
                row = prices * utils.get_weighted_number_of_shares_to_buy(prices, weights, starting_capital)
                portfolio_cum_ret.loc[day, :] = row
                available_money.loc[day] = available_money.loc[day] - (row.sum())
            else:
                available_money.loc[day] = available_money.loc[current_day]

            # Change current day
            current_day = day

        self.history['available money'] = available_money
        self.history['portfolio cumulative returns'] = portfolio_cum_ret
        self.history['weights'] = pd.DataFrame(weights_dictionary).T
        self.history['strategy cumulative returns'] = portfolio_cum_ret.sum(axis=1)
        return portfolio_cum_ret.sum(axis=1)


class GlobalMinimumVariance:
    def __init__(self):
        self.history = dict()

    @staticmethod
    def compute_weights(covariance_matrix: pd.DataFrame) -> pd.Series:
        n = covariance_matrix.shape[0]
        weights = cv.Variable(n, nonneg=True)
        portfolio_volatility = cv.quad_form(weights, covariance_matrix)
        problem = cv.Problem(
            cv.Minimize(portfolio_volatility),
            [
                weights.sum() == 1,
                weights >= 0
            ]
        )
        problem.solve()
        return pd.Series(weights.value, index=covariance_matrix.index)

    def strategy_simulation(self, stocks_data: pd.DataFrame,
                            starting_day: str | pd.Timestamp,
                            starting_capital: float,
                            volatility_window: int = 504,
                            rebalance_frequency: int = 30) -> pd.Series:

        weights_dictionary = dict()

        # Select all days
        days = stocks_data.loc[starting_day:].index
        current_day = days[0]

        # Initialize dataframe
        portfolio_cum_ret = pd.DataFrame(index=days, columns=stocks_data['Close'].columns)

        # Initialize series to keep track of available cash
        available_money = pd.Series(index=days)

        # Compute returns and set parameters for the loop
        stocks_adj_returns = utils.compute_returns(stocks_data['Adj Close'])
        last_rebalance_day = current_day

        # Compute covariance matrix and weights
        cov_matrix = stocks_adj_returns.loc[: current_day].iloc[-volatility_window:].cov()
        weights = self.compute_weights(cov_matrix)
        weights_dictionary[current_day] = weights

        # Add first row
        prices = stocks_data["Close"].loc[current_day]
        row = prices * utils.get_weighted_number_of_shares_to_buy(prices, weights, starting_capital)
        portfolio_cum_ret.loc[current_day, :] = row

        # Update available money
        available_money.loc[current_day] = starting_capital - row.sum()

        for day in tqdm(days[1:]):
            # Update cum returns
            stocks_returns = stocks_adj_returns.loc[day]
            portfolio_cum_ret.loc[day, :] = portfolio_cum_ret.loc[current_day, :] * (stocks_returns + 1)

            # Handle period change
            if (day - last_rebalance_day).days > rebalance_frequency:
                last_rebalance_day = day

                # Sell all stocks
                available_money.loc[day] = available_money.loc[current_day] + portfolio_cum_ret.loc[day].sum()

                # Compute volatility and weights
                cov_matrix = stocks_adj_returns.loc[: current_day].iloc[-volatility_window:].cov()
                weights = self.compute_weights(cov_matrix)
                weights_dictionary[current_day] = weights

                # Buy stocks
                prices = stocks_data["Close"].loc[day]
                row = prices * utils.get_weighted_number_of_shares_to_buy(prices, weights, starting_capital)
                portfolio_cum_ret.loc[day, :] = row
                available_money.loc[day] = available_money.loc[day] - (row.sum())
            else:
                available_money.loc[day] = available_money.loc[current_day]

            # Change current day
            current_day = day

        self.history['available money'] = available_money
        self.history['portfolio cumulative returns'] = portfolio_cum_ret
        self.history['weights'] = pd.DataFrame(weights_dictionary).T
        self.history['strategy cumulative returns'] = portfolio_cum_ret.sum(axis=1)
        return portfolio_cum_ret.sum(axis=1)


class MeanVariance:
    def __init__(self):
        self.history = dict()

    @staticmethod
    def compute_weights(returns, mu_star) -> pd.Series:
        mu = returns.mean()
        covariance_matrix = returns.cov()
        n = covariance_matrix.shape[0]

        weights = cv.Variable(n)

        portfolio_variance = cv.quad_form(weights, covariance_matrix)

        problem = cv.Problem(
            cv.Minimize(portfolio_variance),
            [
                weights.sum() == 1,
                # weights >= 0,
                weights @ mu >= mu_star
            ]
        )
        problem.solve()
        return pd.Series(weights.value, index=covariance_matrix.index)

    def compute_efficient_frontieer(self, returns, max_mu: float = 0.002) -> (np.ndarray, np.ndarray):
        mu_star = np.linspace(0, max_mu, 20)
        standard_deviations = []
        for mu in mu_star:
            weights = self.compute_weights(returns, mu)
            portfolio_sd = np.sqrt(weights.T @ returns.cov() @ weights)
            standard_deviations.append(portfolio_sd)
        return np.array(standard_deviations), np.array(mu_star)

    def strategy_simulation(self, stocks_data: pd.DataFrame,
                            starting_day: str | pd.Timestamp,
                            starting_capital: float,
                            mu_star: float,
                            volatility_window: int = 504,
                            rebalance_frequency: int = 30) -> pd.Series:

        weights_dictionary = dict()

        # Select all days
        days = stocks_data.loc[starting_day:].index
        current_day = days[0]

        # Initialize dataframe
        portfolio_cum_ret = pd.DataFrame(index=days, columns=stocks_data['Close'].columns)

        # Initialize series to keep track of available cash
        available_money = pd.Series(index=days)

        # Compute returns and set parameters for the loop
        stocks_adj_returns = utils.compute_returns(stocks_data['Adj Close'])
        last_rebalance_day = current_day

        # Compute covariance matrix and weights
        weights = self.compute_weights(stocks_adj_returns.loc[: current_day].iloc[-volatility_window:], mu_star)
        weights_dictionary[current_day] = weights

        # Add first row
        prices = stocks_data["Close"].loc[current_day]
        row = prices * utils.get_weighted_number_of_shares_to_buy(prices, weights, starting_capital)
        portfolio_cum_ret.loc[current_day, :] = row

        # Update available money
        available_money.loc[current_day] = starting_capital - row.sum()

        for day in tqdm(days[1:]):
            # Update cum returns
            stocks_returns = stocks_adj_returns.loc[day]
            portfolio_cum_ret.loc[day, :] = portfolio_cum_ret.loc[current_day, :] * (stocks_returns + 1)

            # Handle period change
            if (day - last_rebalance_day).days > rebalance_frequency:
                last_rebalance_day = day

                # Sell all stocks
                available_money.loc[day] = available_money.loc[current_day] + portfolio_cum_ret.loc[day].sum()

                # Compute volatility and weights
                weights = self.compute_weights(stocks_adj_returns.loc[: current_day].iloc[-volatility_window:], mu_star)
                weights_dictionary[current_day] = weights

                # Buy stocks
                prices = stocks_data["Close"].loc[day]
                row = prices * utils.get_weighted_number_of_shares_to_buy(prices, weights, starting_capital)
                portfolio_cum_ret.loc[day, :] = row
                available_money.loc[day] = available_money.loc[day] - (row.sum())
            else:
                available_money.loc[day] = available_money.loc[current_day]

            # Change current day
            current_day = day

        self.history['available money'] = available_money
        self.history['portfolio cumulative returns'] = portfolio_cum_ret
        self.history['weights'] = pd.DataFrame(weights_dictionary).T
        self.history['strategy cumulative returns'] = portfolio_cum_ret.sum(axis=1)
        return portfolio_cum_ret.sum(axis=1)
