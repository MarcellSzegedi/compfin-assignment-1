"""Collects commands in the hedging directory."""

import typer

from .diff_vol import app as diff_vol_app
from .same_vol import app as same_vol_app

app = typer.Typer()

app.add_typer(same_vol_app)
app.add_typer(diff_vol_app)
