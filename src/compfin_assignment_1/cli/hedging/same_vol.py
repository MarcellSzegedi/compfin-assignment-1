"""Plotting scripts for hedging simulation with same volatility used for assets and hedging."""

import logging
from typing import Annotated, Dict, Union

import numpy as np
import numpy.typing as npt
import typer
from matplotlib import pyplot as plt

from compfin_assignment_1.cli.common import (
    calculate_confidence_intervals_hedge_error,
    plot_hedge_discrepancy,
)
from compfin_assignment_1.delta_hedge import OptionHedging
from compfin_assignment_1.utils.logging_config import setup_logging, simulation_progress_logging

setup_logging()
logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command(name="hedging-with-same-vol")
def plot_hedging_result():
    """Plots and saves the hedging strategy and the option price."""
    market_settings = {
        "s_0": 100,
        "r": 0.01,
        "strike": 99,
        "t_end": 1,
        "volatility_c": 0.2,
        "random_seed": 100,
    }

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(30, 10), constrained_layout=True)

    hedging_freq = ["daily", "weekly", "monthly"]

    for i in range(9):
        row_idx = i % 3
        col_idx = i // 3
        axis = ax[row_idx, col_idx]

        curr_hedging_freq = hedging_freq[row_idx]

        hedging_port_price, option_price, stock_price = (
            OptionHedging.simulate_hedging_strategy_w_fixed_vol_one_year(
                hedging_freq=curr_hedging_freq, **market_settings
            )
        )

        t = np.linspace(0, market_settings["t_end"], hedging_port_price.shape[0] - 1)
        axis.plot(
            t,
            hedging_port_price[:-1],
            label="Hedging Portfolio",
            color="red",
            linestyle="--",
            linewidth=2,
        )
        axis.plot(t, option_price[:-1], label="Option Price", color="blue", linewidth=0.5)
        axis.set_ylabel("Hedging Portfolio / Option Price in $")

        ax_twin = axis.twinx()
        ax_twin.plot(t, stock_price[:-1], label="Stock Price", color="green", linewidth=0.5)
        ax_twin.set_ylabel("Stock Price in $")

        axis.set_title(f"Hedging frequency is {curr_hedging_freq}")
        axis.set_xlabel("Time")
        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axis.legend(lines1 + lines2, labels1 + labels2)

    fig.suptitle("European Call Option Replication Over One Year Using Delta Hedging", fontsize=16)

    plt.savefig("results/figures/option_vs_hedging_replication_same_vol_NEW.png", dpi=600)
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
    market_settings = {"s_0": 100, "r": 0.01, "strike": 99, "t_end": 1, "volatility_c": 0.2}
    hedging_freq_param = ["daily", "weekly", "monthly"]

    discrepancy_trajectories = {
        hedging_freq: simulate_hedging_port_discrepancies(hedging_freq, market_settings, n_sim)
        for hedging_freq in hedging_freq_param
    }

    mean_discrepancy, conf_int_upper_bound, conf_int_lower_bound = (
        calculate_confidence_intervals_hedge_error(discrepancy_trajectories, n_sim, alpha)
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
