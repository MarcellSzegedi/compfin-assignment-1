"""Collects commands in the hedging directory."""

import typer

from .same_vol import app as same_vol_app

app = typer.Typer()

app.add_typer(same_vol_app)