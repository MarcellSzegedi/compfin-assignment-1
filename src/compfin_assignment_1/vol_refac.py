"""Market volatility calculations."""

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view
from skfolio.datasets import load_sp500_implied_vol_dataset
from utils.param_validation import (
    check_rolling_window_length as check_window_length,
)
from utils.param_validation import (
    check_volatility_class_parameter as check_input_params,
)


class VolatilityCalculations:
    """Contains methods to calculate, estimate and compare market volatility."""

    def __init__(self, ticker: str, startdate: datetime, enddate: Union[datetime, str]):
        """Initialises VolatilityCalculations instance."""
        check_input_params(ticker, startdate, enddate)

        self._ticker = ticker
        self._startdate = startdate.replace(hour=0, minute=0, second=0, microsecond=0)
        self._enddate = (
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) if enddate == "now" else enddate
        )

        self.market_data = self._parse_stock_data()
        self.closing_prices = np.array(self.market_data["Close"])
        self.returns = self._calc_market_returns()
        self.high_prices = np.array(self.market_data["High"])
        self.low_prices = np.array(self.market_data["Low"])

    @classmethod
    def compute_vol_rolling_window(
        cls,
        ticker: str,
        startdate: datetime,
        enddate: Union[datetime, str],
        window_length: int = 30,
        return_implied: bool = False,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
        """Calculates the realised and parkinson volatility for every window.

        Args:
            window_length: The length of the window.
            ticker: Stock of interest.
            startdate: Starting date of the analysis of the market data.
            enddate: Ending date of the analysis of the market data.
            return_implied: Boolean to choose whether to return implied volatility.

        Returns:
            Volatility time series for both methodology. (1D numpy array)
        """
        vol_instance = cls(ticker, startdate, enddate)
        check_window_length(window_length, len(vol_instance.returns))

        returns_window = sliding_window_view(vol_instance.returns, window_length)
        high_prices_windows = sliding_window_view(vol_instance.high_prices, window_length)[:-1]
        low_prices_windows = sliding_window_view(vol_instance.low_prices, window_length)[:-1]

        realised_vol = vol_instance.realised_vol_calc_roll_wind(returns_window)
        parkinson_vol = vol_instance.parkinson_volatility_calc_roll_win(high_prices_windows, low_prices_windows)

        implied_vol = None
        if return_implied:
            start = vol_instance._startdate
            end = vol_instance._enddate - timedelta(days=1)
            implied_vol = load_sp500_implied_vol_dataset()[start:end]
            implied_vol = np.array(implied_vol[vol_instance._ticker])[:-window_length]

        return realised_vol, parkinson_vol, implied_vol

    @staticmethod
    def parkinson_volatility_calc_roll_win(
        high_prices: npt.NDArray[np.float64], low_prices: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculates parkinson volatility for multiple windows.

        Args:
            high_prices: 2D numpy array, where every row represents a window of high
                            prices of the day
            low_prices: 2D numpy array, where every row represents a window of low
                            prices of the day

        Returns:
            Parkinson volatility time series. (1D numpy array)
        """
        inside_sum = np.power(np.log(high_prices / low_prices), 2)
        parkinson_scaler = 1 / (4 * np.log(2))
        return np.sqrt(parkinson_scaler * np.sum(inside_sum, axis=1))

    @staticmethod
    def realised_vol_calc_roll_wind(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculates realised volatility for multiple windows.

        Args:
            returns: 2D numpy array, where every row represents a window of
                                returns

        Returns:
            Realised volatility time series. (1D numpy array)
        """
        realised_mu = np.mean(returns, axis=1)
        scaler = 1 / (len(returns) - 1)
        vol_squared = scaler * np.sum(np.power(returns - realised_mu.reshape(-1, 1), 2), axis=1)
        return np.sqrt(vol_squared)

    def _parse_stock_data(self) -> pd.DataFrame:
        """Parses the stock price data given the ticker and the start and end date."""
        stock_data = yf.download(self._ticker, start=self._startdate, end=self._enddate)

        stock_data.columns = stock_data.columns.droplevel(1)

        return stock_data

    def _calc_market_returns(self):
        """Calculates market returns using only the closing prices."""
        return (self.closing_prices[1:] - self.closing_prices[:-1]) / self.closing_prices[:-1]


real_vol, park_vol, implied_vol = VolatilityCalculations.compute_vol_rolling_window(
    "AAPL", startdate=datetime(2021, 1, 20), enddate=datetime(2021, 4, 6), window_length=2, return_implied=True
)
