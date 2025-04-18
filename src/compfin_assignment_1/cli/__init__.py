"""Main entry point for the CLI application."""

import typer

from .hedging import app as hedging_app

app = typer.Typer()

app.add_typer(hedging_app, name="hedging")
