"""Plotting scripts for hedging simulation with same volatility used for assets and hedging."""

from typing import Annotated

import numpy as np
import typer
from matplotlib import pyplot as plt

from compfin_assignment_1.delta_hedge import OptionHedging

app = typer.Typer()


@app.command(name="hedging-with-same-vol")
def main(
    hedging_freq: Annotated[
        str,
        typer.Option(
            "--hedging-freq",
            help="Frequency of the update of the hedging portfolio.",
        ),
    ],
):
    """Plots and saves the hedging strategy and the option price."""
    market_settings = {"s_0": 100, "r": 0.01, "strike": 99, "t_end": 1, "volatility": 0.2}

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
