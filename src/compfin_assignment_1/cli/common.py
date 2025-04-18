"""Common function used for cli commands."""

from typing import Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.stats as st
from matplotlib import pyplot as plt


def calculate_confidence_intervals_hedge_error(
    discrepancy_trajectories: Dict[Union[str, float], npt.NDArray[np.float64]],
    n_sim: int,
    alpha: float,
) -> Tuple[
    Dict[Union[str, float], npt.NDArray[np.float64]],
    Dict[Union[str, float], npt.NDArray[np.float64]],
    Dict[Union[str, float], npt.NDArray[np.float64]],
]:
    """Calculates the upper and lower bound of the confidence intervals around the mean values.

    The calculation is done for every time point for every hedge frequencies.

    Args:
        discrepancy_trajectories: Simulated hedge portfolio discrepancy trajectories for different
                                    hedging frequency parameters. (2D numpy array)
        n_sim: Number of simulated trajectories.
        alpha: Significance level.

    Returns:
        Mean values, and the upper and lower bounds of the confidence intervals.
    """
    hedging_freq_params = list(discrepancy_trajectories.keys())
    mean_discrepancy = {
        hedging_freq: np.mean(np.array(discrepancies), axis=0)
        for hedging_freq, discrepancies in discrepancy_trajectories.items()
    }
    std_error_discrepancy = {
        hedging_freq: np.std(np.array(discrepancies), axis=0, ddof=1) / np.sqrt(n_sim)
        for hedging_freq, discrepancies in discrepancy_trajectories.items()
    }

    upper_bounds = {
        hedging_freq: (
            mean_discrepancy[hedging_freq]
            + st.norm.ppf(1 - alpha / 2) * std_error_discrepancy[hedging_freq]
        )
        for hedging_freq in hedging_freq_params
    }
    lower_bounds = {
        hedging_freq: (
            mean_discrepancy[hedging_freq]
            - st.norm.ppf(1 - alpha / 2) * std_error_discrepancy[hedging_freq]
        )
        for hedging_freq in hedging_freq_params
    }

    return mean_discrepancy, upper_bounds, lower_bounds


def plot_hedge_discrepancy(
    mean_values: Dict[str, npt.NDArray[np.float64]],
    upper_bounds: Dict[str, npt.NDArray[np.float64]],
    lower_bounds: Dict[str, npt.NDArray[np.float64]],
    t_end: float,
) -> None:
    """Plots the various confidence intervals for the discrepancy trajectories.

    Args:
        mean_values: Mean discrepancy values for different hedging portfolio frequencies.
        upper_bounds: Upper bound of the confidence intervals for different hedging portfolio
                        frequencies.
        lower_bounds: Lower bound of the confidence intervals for different hedging portfolio
                        frequencies.
        t_end: End time of the simulated trajectories in year.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    colors = ["red", "blue", "green", "purple", "orange", "green"]

    for i, hedging_freq in enumerate(mean_values.keys()):
        t = np.linspace(0, t_end, mean_values[hedging_freq].shape[0])

        plt.plot(t, mean_values[hedging_freq], linewidth=1.5, label=hedging_freq, color=colors[i])
        plt.plot(t, upper_bounds[hedging_freq], linewidth=0.3, linestyle="--", color=colors[i])
        plt.plot(t, lower_bounds[hedging_freq], linewidth=0.3, linestyle="--", color=colors[i])

        plt.fill_between(
            t, lower_bounds[hedging_freq], upper_bounds[hedging_freq], alpha=0.1, color=colors[i]
        )

    plt.title("Delta Hedging Error from the Option Price")
    plt.xlabel("Time")
    plt.ylabel("Delta Hedging Error ($)")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("results/figures/hedging_replication_error.png", dpi=600)
    plt.show()
