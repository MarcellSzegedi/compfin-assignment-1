"""Plotting scripts for hedging simulation with same volatility used for assets and hedging."""

import typer

app = typer.Typer()


@app.command(name="same-vol")
def same_vol():
    """Example for CLI usage."""
    print("It is working!")
