"""Plotting scripts for hedging simulation with same volatility used for assets and hedging."""

import logging
from typing import Annotated, Dict, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.stats as st
import typer
from matplotlib import pyplot as plt

from compfin_assignment_1.delta_hedge import OptionHedging
from compfin_assignment_1.utils.logging_config import setup_logging, simulation_progress_logging

setup_logging()
logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command(name="hedging-with-same-vol")
def plot_hedging_result(
    hedging_freq: Annotated[
        str,
        typer.Option(
            "--hedging-freq",
            help="Frequency of the update of the hedging portfolio.",
        ),
    ],
):
    """Plots and saves the hedging strategy and the option price."""
    market_settings = {
        "s_0": 100,
        "r": 0.01,
        "strike": 99,
        "t_end": 1,
        "volatility": 0.2,
        "random_seed": 100,
    }

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    for i in range(6):
        row_idx = i % 2
        col_idx = i // 2
        axis = ax[row_idx, col_idx]

        hedging_port_price, option_price, stock_price = (
            OptionHedging.simulate_hedging_strategy_w_fixed_vol_one_year(
                hedging_freq=hedging_freq, **market_settings
            )
        )

        t = np.linspace(0, market_settings["t_end"], hedging_port_price.shape[0])
        axis.plot(
            t,
            hedging_port_price,
            label="Hedging Portfolio",
            color="red",
            linestyle="--",
            linewidth=2,
        )
        axis.plot(t, option_price, label="Option Price", color="blue", linewidth=0.5)
        axis.set_ylabel("Hedging Portfolio / Option Price in $")

        ax_twin = axis.twinx()
        ax_twin.plot(t, stock_price, label="Stock Price", color="green", linewidth=0.5)
        ax_twin.set_ylabel("Stock Price in $")

        axis.set_xlabel("Time")
        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axis.legend(lines1 + lines2, labels1 + labels2)

    fig.suptitle("European Call Option Replication Over One Year Using Delta Hedging", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("results/figures/option_vs_hedging_replication.png", dpi=600)
    plt.show()


@app.command(name="hedging-error")
def plot_hedging_error(
    n_sim: Annotated[
        int, typer.Option("--n-sim", help="Number of simulated trajectories.", min=1)
    ],
    alpha: Annotated[
        float, typer.Option("--alpha", help="Significance level.", min=0, max=1)
    ] = 0.05,
):
    """Plots and saves the discrepancy of the hedging strategy from the option price.

    The discrepancy is simulated for various delta hedging strategies update parameter.

    Args:
        n_sim: Number of simulated trajectories.
        alpha: Significance level.

    Returns:
        None
    """
    market_settings = {"s_0": 100, "r": 0.01, "strike": 99, "t_end": 1, "volatility": 0.2}
    hedging_freq_param = ["daily", "weekly", "monthly"]

    discrepancy_trajectories = {
        hedging_freq: simulate_hedging_port_discrepancies(hedging_freq, market_settings, n_sim)
        for hedging_freq in hedging_freq_param
    }

    mean_discrepancy, conf_int_upper_bound, conf_int_lower_bound = calculate_confidence_intervals(
        discrepancy_trajectories, n_sim, alpha
    )

    plot_hedge_discrepancy(
        mean_discrepancy, conf_int_upper_bound, conf_int_lower_bound, market_settings["t_end"]
    )


def simulate_hedging_port_discrepancies(
    hedging_freq: str, model_parameters: Dict[str, Union[float, int]], n_sim: int
) -> npt.NDArray[np.float64]:
    """Calculates 'n_sim' number of discrepancy trajectories.

    Args:
        hedging_freq: Frequency of the update of the hedging portfolio.
        model_parameters: Rest of the model parameters.
        n_sim: Number of simulated trajectories.

    Returns:
        Simulated trajectories. (2D numpy array)
    """
    discrepancy_realisation = []
    for i in range(n_sim):
        discrepancy_realisation.append(
            OptionHedging.hedging_discrepancy_simulation(
                hedging_freq=hedging_freq, **model_parameters
            )
        )
        simulation_progress_logging(logger, "Simulation of delta hedging discrepancy", i, 10)

    return np.array(discrepancy_realisation)


def calculate_confidence_intervals(
    discrepancy_trajectories: Dict[str, npt.NDArray[np.float64]],
    n_sim: int,
    alpha: float,
) -> Tuple[
    Dict[str, npt.NDArray[np.float64]],
    Dict[str, npt.NDArray[np.float64]],
    Dict[str, npt.NDArray[np.float64]],
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
    plt.figure(figsize=(20, 10))
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

    plt.savefig("results/figures/hedging_replication_error.png", dpi=600)
    plt.show()
