"""Plotting scripts for hedging simulation with diff. volatility used for assets and hedging."""

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


@app.command(name="hedging-with-diff-vol")
def hedging_with_diff_vol():
    """Plots and saves the hedging strategy and the option price calculated with diff. vol.."""
    market_settings = {
        "s_0": 100,
        "r": 0.01,
        "strike": 99,
        "t_end": 1,
        "volatility_c": 0.2,
        "hedging_freq": "weekly",
    }

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(30, 10), constrained_layout=True)
    volatility_h_params = [0.05, 0.2, 0.5]

    for i in range(9):
        row_idx = i % 3
        col_idx = i // 3
        axis = ax[row_idx, col_idx]

        curr_hedging_vol = volatility_h_params[row_idx]

        hedging_port_price, option_price, stock_price = (
            OptionHedging.simulate_hedging_strategy_w_fixed_vol_one_year(
                volatility_h=curr_hedging_vol, random_seed=col_idx, **market_settings
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

        axis.set_title(f"Volatility Used for Hedging: {curr_hedging_vol}")
        axis.set_xlabel("Time")
        lines1, labels1 = axis.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axis.legend(lines1 + lines2, labels1 + labels2)

    fig.suptitle("European Call Option Replication Over One Year Using Delta Hedging", fontsize=16)

    plt.savefig("results/figures/option_vs_hedging_replication_diff_vol_NEW.png", dpi=600)
    plt.show()


@app.command(name="hedging-error-diff-vol")
def plot_hedging_error_with_diff_vol(
    n_sim: Annotated[
        int, typer.Option("--n-sim", help="Number of simulated trajectories.", min=1)
    ],
    alpha: Annotated[
        float, typer.Option("--alpha", help="Significance level.", min=0, max=1)
    ] = 0.05,
):
    """Plots and saves the discrepancy of the hedging strategy from the option price.

    The discrepancy is simulated for various delta hedging volatility parameter.

    Args:
        n_sim: Number of simulated trajectories.
        alpha: Significance level.

    Returns:
        None
    """
    market_settings = {
        "s_0": 100,
        "r": 0.01,
        "strike": 99,
        "t_end": 1,
        "volatility_c": 0.2,
        "hedging_freq": "weekly",
    }
    volatility_h_params = [0.05, 0.2, 0.5]

    discrepancy_trajectories = {
        vol_h: simulate_hedging_port_discrepancies_with_diff_vol(vol_h, market_settings, n_sim)
        for vol_h in volatility_h_params
    }

    mean_discrepancy, conf_int_upper_bound, conf_int_lower_bound = (
        calculate_confidence_intervals_hedge_error(discrepancy_trajectories, n_sim, alpha)
    )

    plot_hedge_discrepancy(
        mean_discrepancy, conf_int_upper_bound, conf_int_lower_bound, market_settings["t_end"]
    )


def simulate_hedging_port_discrepancies_with_diff_vol(
    volatility_h: float, model_parameters: Dict[str, Union[float, int]], n_sim: int
) -> npt.NDArray[np.float64]:
    """Calculates 'n_sim' number of discrepancy trajectories.

    Args:
        volatility_h: Volatility parameter used for hedging.
        model_parameters: Rest of the model parameters.
        n_sim: Number of simulated trajectories.

    Returns:
        Simulated trajectories. (2D numpy array)
    """
    discrepancy_realisation = []
    for i in range(n_sim):
        discrepancy_realisation.append(
            OptionHedging.hedging_discrepancy_simulation(
                volatility_h=volatility_h, **model_parameters
            )
        )
        simulation_progress_logging(logger, "Simulation of delta hedging discrepancy", i, 10)

    return np.array(discrepancy_realisation)
