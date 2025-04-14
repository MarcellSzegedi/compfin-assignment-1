"""Methods that calculates the optimal hedging strategy for the european option."""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.stats as st

from compfin_assignment_1.utils.constants import FREQUENCY_TO_HOUR, N_HOURS_PER_YEAR


class OptionHedging:
    """Simulates the pricing and delta hedging of a European option.

    This class provides methods to generate price paths for the underlying asset, compute the
    European option's price using a specified pricing model (using Black-Scholes formula), and
    simulate a delta hedging strategy over time.
    """

    def __init__(
        self,
        s_o: float,
        r: float,
        strike: float,
        n_step: int,
        t_end: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the option hedging instance.

        Args:
            s_o: Stock price at time 0.
            r: Yearly risk-free interest rate.
            strike: Strike price of the european option.
            n_step: Number of dividing time intervals of the [0, T_end] time span.
            t_end: End time of the simulation in years.
            random_seed: Random seed for the Random Number Generator.
        """
        self.s_o = s_o
        self.r = r * t_end / n_step
        self.strike = strike
        self.t_end = t_end
        self.n_step = n_step
        self.step_size = t_end / n_step
        self.time_remaining = np.append(
            t_end - np.arange(n_step) * self.step_size, np.array([0.01 * self.step_size])
        )
        self._rng = np.random.default_rng(random_seed)

    @classmethod
    def simulate_hedging_strategy_w_fixed_vol_one_year(
        cls,
        s_0: float,
        r: float,
        strike: float,
        t_end: float,
        volatility: float,
        hedging_freq: str,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculates hedge amounts in asset and cash for a European option.

        The simulation is performed by the following process:
        (1) Choose an appropriately small timestep
        (2) Simulate the time series of the price of the underlying asset using Euler method
                for the whole interval [0, T]
        (3) Calculate the option price
        (4) Calculate the amount to be held of the asset and cash to follow the delta hedging
                strategy

        Args:
            s_0: Stock price at time 0.
            r: Yearly risk-free interest rate.
            strike: Strike price  of the european option.
            t_end: End time of the simulation in years.
            volatility: Time independent volatility used for the underlying asset price simulation
                            and the delta hedge calculation.
            hedging_freq: Frequency of the hedging strategy.

        Returns:
            Hedging portfolio value for every time point (1D numpy array)
            European option price in time (1D numpy array)
            Underlying asset price in time (1D numpy array)
        """
        n_time_step = N_HOURS_PER_YEAR * t_end
        hedging_instance = cls(s_0, r, strike, n_time_step, t_end)

        stock_prices = hedging_instance.stock_price_simulation(volatility)
        option_prices = hedging_instance.option_price_calculation(stock_prices, volatility)

        asset_value, cash_values = hedging_instance.hedge_coefficients_calculation(
            option_prices, stock_prices, volatility, hedging_freq
        )

        hedging_port_value = hedging_instance.calculate_hedging_port_value(
            asset_value, cash_values, hedging_freq
        )

        return hedging_port_value, option_prices, stock_prices

    def stock_price_simulation(self, volatility: float) -> npt.NDArray[np.float64]:
        """Simulates a time-series for the stock price over the interval [0, T_end].

        The simulation is based on the stock price dynamics assumed in the Black-Scholes model.

        Args:
            volatility: Volatility of the underlying stock.

        Returns:
            Time-series of the prices (1D numpy array).
        """
        wiener_increments = self._rng.normal(loc=0, scale=1, size=self.n_step)
        stochastic_term = wiener_increments * volatility * np.sqrt(self.step_size)
        drift_term = self.r * self.step_size
        final_increments = 1.0 + drift_term + stochastic_term

        return np.concatenate(
            (np.array([self.s_o]), self.s_o * np.cumprod(final_increments)), axis=0
        )

    def option_price_calculation(
        self, stock_price: npt.NDArray[np.float64], volatility: float
    ) -> npt.NDArray[np.float64]:
        """Calculates the price of a European call option using the Black-Scholes formula.

        Args:
            stock_price: Array of stock prices at each time step.
            volatility: Volatility of the underlying stock.

        Returns:
            Array of option prices corresponding to each stock price in the input. (1D numpy array)
        """
        d1_values = (
            np.log(stock_price / self.strike) + (self.r + volatility**2 / 2) * self.time_remaining
        ) / (volatility * np.sqrt(self.time_remaining))
        d2_values = d1_values - volatility * np.sqrt(self.time_remaining)

        n_d1 = st.norm.cdf(d1_values)
        n_d2 = st.norm.cdf(d2_values)

        return stock_price * n_d1 - np.exp(-self.r * self.time_remaining) * self.strike * n_d2

    def hedge_coefficients_calculation(
        self,
        option_price: npt.NDArray[np.float64],
        stock_price: npt.NDArray[np.float64],
        volatility: float,
        hedging_freq: str,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculates the amount needed from the cask and the stock to replicate the option.

        The function assumes that the stock and option prices are calculated in every hour.

        The hedging follows the following procedure:
        (1) Determine the information of interest based on the hedging portfolio update frequency
        (2) Calculate the optimal amount to be held of the underlying asset in every update time
                point. In case of the European option following the black-scholes assumptions the
                delta value is N(d_1)
        (3) Calculate the difference in the price kept in the asset for every time point
                For example: PriceDiff_t = (N(d_1)_t - N(d_1)_t-1) * S_t
                The Price_diff is going to be equal to the amount that should be invested (in case
                its negative then shorting) cash at time 't'
        (4) Repeat steps (2) - (3) for every timepoint

        Args:
            option_price: Array of option prices at each time step.
            stock_price: Array of stock prices at each time step.
            volatility: Volatility of the underlying stock.
            hedging_freq: Hedging frequency.

        Returns:
            Dynamic amounts to held from the underlying asset and cash to hedge the option.
            (1D numpy array)
        """
        hedging_update_indices, time_step = self._determine_hedging_update_times(hedging_freq)

        hedge_port_asset, hedge_port_cash, delta_values = self._initialise_hedging_portfolio(
            stock_price, option_price, volatility
        )

        for i in hedging_update_indices[1:]:
            delta = self._delta_calculation(stock_price[i], self.time_remaining[i], volatility)

            asset_value_diff = (delta - delta_values[-1]) * stock_price[i]
            cash_value = asset_value_diff + hedge_port_cash[-1] * np.exp(self.r * time_step)

            hedge_port_asset.append(delta * stock_price[i])
            hedge_port_cash.append(cash_value)
            delta_values.append(delta)

        return np.array(hedge_port_asset), np.array(hedge_port_cash)

    def calculate_hedging_port_value(
        self,
        stock_value: npt.NDArray[np.float64],
        cash_value: npt.NDArray[np.float64],
        hedging_freq: str,
    ) -> npt.NDArray[np.float64]:
        """Calculates the value of the hedging strategy at every time point.

        For the in-between points the function uses linear interpolation.

        Args:
            stock_value: Value kept in stock for every time point, where the hedging strategy is
                            updated.
            cash_value: Value kept in cash for every time point, where the hedging strategy is
                            updated.
            hedging_freq: Hedging frequency.

        Returns:
            Cumulated value of the hedging strategy for every time point. (1D numpy array)
        """
        hedging_portfolio_val = stock_value - cash_value
        hedging_update_indices, _ = self._determine_hedging_update_times(hedging_freq)

        hedging_time_points = self.time_remaining[hedging_update_indices]

        # Order of the time series needs to be reversed
        # due to the function configuration of np.interp
        interpolated_hedging_port = np.interp(
            x=self.time_remaining[::-1],
            xp=hedging_time_points[::-1],
            fp=hedging_portfolio_val[::-1],
        )
        return interpolated_hedging_port[::-1]

    def _initialise_hedging_portfolio(
        self,
        stock_price: npt.NDArray[np.float64],
        option_price: npt.NDArray[np.float64],
        volatility: float,
    ) -> Tuple[list, list, list]:
        """Initialises a portfolio that delta hedges an european call option.

        Args:
            stock_price: Stock prices at each time point.
            option_price: Option prices at each time point.
            volatility: Volatility of the underlying stock.

        Returns:
            Three lists containing the optimal amount (according to the delta hedging strategy) of
            asset, cash value, and asset amount, respectively.
        """
        asset_start, cash_start, delta = self._hedging_port_start_calculation(
            stock_price, option_price, volatility
        )
        return [asset_start], [cash_start], [delta]

    def _hedging_port_start_calculation(
        self,
        stock_price: npt.NDArray[np.float64],
        option_price: npt.NDArray[np.float64],
        volatility: float,
    ) -> Tuple[float, float, float]:
        """Calculates the starting asset and cash amount for the call option.

        Args:
            stock_price: Array of stock prices at each time step.
            option_price: Array of option prices at each time step.
            volatility: Volatility of the underlying stock.

        Returns:
            Starting values for the asset, cash values, and asset amount, respectively.
        """
        delta = self._delta_calculation(
            float(stock_price[0]), float(self.time_remaining[0]), volatility
        )

        asset_value = float(stock_price[0]) * delta
        cash_value = asset_value - float(option_price[0])

        return asset_value, cash_value, delta

    def _delta_calculation(
        self, stock_price: float, time_remaining: float, volatility: float
    ) -> float:
        """Calculates the delta parameter for a single time point.

        The delta parameter is the rate of change of the option price with respect to changes in
        the price of the underlying asset.

        Args:
            stock_price: Stock price.
            time_remaining: Time remaining until the maturity of the option contract.
            volatility: Volatility of the underlying stock.

        Returns:
            Delta parameter.
        """
        d1 = (
            np.log(stock_price / self.strike) + (self.r + volatility**2 / 2) * time_remaining
        ) / (volatility * np.sqrt(time_remaining))
        return st.norm.cdf(d1)

    def _determine_hedging_update_times(
        self,
        hedging_freq: str,
    ) -> Tuple[npt.NDArray[np.int64], float]:
        """Determines the indexes on [0, T_end] at which the hedging strategy is updated.

        Args:
            hedging_freq: Hedging frequency.

        Returns:
            Indices of updates (1D numpy array) and the time step between two consecutive updates.
            (float)
        """
        match hedging_freq:
            case "hourly":
                return self._compute_update_indices_and_time_step(FREQUENCY_TO_HOUR["hour"])
            case "daily":
                return self._compute_update_indices_and_time_step(FREQUENCY_TO_HOUR["day"])
            case "weekly":
                return self._compute_update_indices_and_time_step(FREQUENCY_TO_HOUR["week"])
            case "biweekly":
                return self._compute_update_indices_and_time_step(FREQUENCY_TO_HOUR["biweek"])
            case "monthly":
                return self._compute_update_indices_and_time_step(FREQUENCY_TO_HOUR["month"])
            case _:
                raise ValueError(
                    "Invalid hedging frequency.Options are: "
                    "- 'hourly' "
                    "- 'daily' "
                    "- 'weekly' "
                    "- 'bi-weekly' "
                    "- 'monthly'."
                )

    def _compute_update_indices_and_time_step(
        self,
        freq: int,
    ) -> Tuple[npt.NDArray[np.int64], float]:
        """Computes the indexes on the [0, T_end] at which the hedging strategy is updated.

        The function assumes that the stock prices are calculated in every hour.

        Args:
            freq: Number of hours to update the hedging portfolio.

        Returns:
            Indices of updates (1D numpy array) and the time step between two consecutive updates.
            (float)
        """
        return np.arange(self.n_step + 1)[::freq], self.step_size * freq
