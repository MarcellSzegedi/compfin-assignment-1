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
        volatility_c: float,
        hedging_freq: str,
        volatility_h: Optional[float] = None,
        random_seed: Optional[int] = None,
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
            volatility_c: Time independent volatility used for the underlying asset price
                            simulation and the european option price calculation.
            hedging_freq: Frequency of the hedging strategy.
            volatility_h: Time independent volatility used for the delta hedging strategy.
            random_seed: Random seed for the Random Number Generator.

        Returns:
            Hedging portfolio value for every time point (1D numpy array)
            European option price in time (1D numpy array)
            Underlying asset price in time (1D numpy array)
        """
        if volatility_h is None:
            volatility_h = volatility_c
        n_time_step = N_HOURS_PER_YEAR * t_end

        hedging_instance = cls(s_0, r, strike, n_time_step, t_end, random_seed)

        stock_prices = hedging_instance.stock_price_simulation(volatility_c)
        option_prices = hedging_instance.option_price_calculation(stock_prices, volatility_c)

        _, cash_value, delta = hedging_instance.hedge_coefficients_calculation(
            option_prices, stock_prices, volatility_h, hedging_freq
        )

        hedging_port_value = hedging_instance.calculate_hedging_port_value(
            delta, stock_prices, cash_value
        )

        return hedging_port_value, option_prices, stock_prices

    @classmethod
    def hedging_discrepancy_simulation(
        cls,
        s_0: float,
        r: float,
        strike: float,
        t_end: float,
        volatility_c: float,
        hedging_freq: str,
        volatility_h: Optional[float] = None,
        random_seed: Optional[int] = None,
    ):
        """Calculates the deviation of the delta hedging portfolio from the european option.

        The function uses the assumptions and the results of the black-scholes model for the
        aforementioned derivative.

        Args:
            s_0: Stock price at time 0.
            r: Yearly risk-free interest rate.
            strike: Strike price  of the european option.
            t_end: End time of the simulation in years.
            volatility_c: Time independent volatility used for the underlying asset price
                            simulation and option price calculation.
            hedging_freq: Frequency of the hedging strategy.
            volatility_h: Time independent volatility used for the delta hedging strategy.
            random_seed: Random seed for the Random Number Generator.

        Returns:
            The difference between the simulated option price and the delta hedging portfolio for
            one trajectory. (1D numpy array)
        """
        if volatility_h is None:
            volatility_h = volatility_c

        hedging_port, option_price, _ = cls.simulate_hedging_strategy_w_fixed_vol_one_year(
            s_0, r, strike, t_end, volatility_c, hedging_freq, volatility_h, random_seed
        )
        return np.abs(hedging_port - option_price) * np.exp(
            -r * np.arange(hedging_port.shape[0]) / hedging_port.shape[0] * t_end
        )

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
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

        return np.array(hedge_port_asset), np.array(hedge_port_cash), np.array(delta_values)

    def calculate_hedging_port_value(
        self,
        stock_amount: npt.NDArray[np.float64],
        stock_price: npt.NDArray[np.float64],
        cash_value: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculates the value of the hedging strategy at every time point.

        Args:
            stock_amount: Amount of stock owned at the hedging portfolio update points.
            stock_price: Array of stock prices at each time point.
            cash_value: Value kept in cash for every time point, where the hedging strategy is
                            updated.

        Returns:
            Cumulated value of the hedging strategy for every time point. (1D numpy array)
        """
        hedging_port_asset_value = self._asset_value_calculation_cont(stock_amount, stock_price)
        hedging_port_cash_value = self._cash_value_calculation_cont(
            cash_value, stock_price.shape[0]
        )

        hedging_portfolio_val = hedging_port_asset_value - hedging_port_cash_value

        return hedging_portfolio_val

    @staticmethod
    def _asset_value_calculation_cont(
        stock_amounts: npt.NDArray[np.float64], stock_price: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculates the value of hedging portfolio asset part for every time point available.

        Args:
            stock_amounts: Amount of stock owned for the update point
            stock_price: Stock prices for every time point available.

        Returns:
            Asset part value of the hedging portfolio for every time point. (2D numpy array)
        """
        asset_amounts = np.repeat(
            stock_amounts[:-1], (len(stock_price) - 1) / (len(stock_amounts) - 1)
        )
        asset_values = asset_amounts * stock_price[:-1]
        asset_values = np.append(asset_values, asset_amounts[-1] * stock_price[-1])
        return asset_values

    def _cash_value_calculation_cont(
        self, cash_value: npt.NDArray[np.float64], total_n_step: int
    ) -> npt.NDArray[np.float64]:
        """Calculates the bank process part value of the hedging portfolio for every time point.

        Args:
            cash_value: Value kept in cash for the hedging portfolio update points.
                            (1D numpy array)
            total_n_step: Number of time step the option simulation is calculated for.

        Returns:
            Bank process values of the hedging portfolio for every time point. (1D numpy array)
        """
        steps = int((total_n_step - 1) / (cash_value.shape[0] - 1))
        cash_value_expanded = np.empty((cash_value.shape[0] - 1) * steps, dtype=np.float64)
        t_values = np.arange(steps)

        for i in range(cash_value.shape[0] - 1):
            start_val = cash_value[i]
            cash_value_expanded[i * steps : (i + 1) * steps] = start_val * np.exp(
                self.r * t_values
            )

        cash_value_expanded = np.append(cash_value_expanded, cash_value[-1])

        return cash_value_expanded

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


# import matplotlib.pyplot as plt
#
# h_port_val, option_price, _ = OptionHedging.simulate_hedging_strategy_w_fixed_vol_one_year(
#     100, 0.01, 99, 1, 0.2, "monthly", 0.2, 100
# )
# plt.figure(figsize=(7, 7))
# plt.plot(h_port_val)
# plt.plot(option_price)
# plt.show()
