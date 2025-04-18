"""Course: Computational Finance
Names: Marcell ..., Michael ... and Tika van Bennekum
Student IDs: ..., ... and 13392425

File description:
    Part 1 of lab assignment 1.
    In this file we determine VIX using the VIX_t estimator.
"""

import datetime
import numpy as np
import pandas as pd
import yfinance as yf


def calculate_F(calls, puts, r, tau):
    """Calculate forward price approximation using put-call parity:
    VIX uses european options, so F ≈ K + e^(rτ) * (C - P)."""
    calls = calls.set_index("strike")
    puts = puts.set_index("strike")

    difference = abs(calls["mid_call"] - puts["mid_put"])
    K_zero = difference.idxmin()

    C = calls.loc[K_zero, "mid_call"]
    P = puts.loc[K_zero, "mid_put"]

    F = K_zero + np.exp(r * tau) * (C - P)
    return F


def calc_delta_K(strikes):
    """Accounts for the fact that not every option contributes
    the same amount."""
    delta_K = np.zeros_like(strikes)
    delta_K[1:-1] = (strikes[2:] - strikes[:-2]) / 2
    delta_K[0] = strikes[1] - strikes[0]
    delta_K[-1] = strikes[-1] - strikes[-2]
    return delta_K


def vix_integral(otm, price_col):
    """Estimates the integral from formula 19."""
    K = otm["strike"].values
    prices = otm[price_col].values
    delta_K = calc_delta_K(K)
    inside_sum = prices * delta_K / (K**2)
    return inside_sum.sum()


def cbeo_vix():
    """The CBOE-quoted VIX."""
    spx_data = yf.download(spx_symbol, start=start_date, end=end_date)
    lastBusDay = spx_data.index[-1]
    vix_data = yf.download(
        "^VIX", start=lastBusDay, end=lastBusDay + datetime.timedelta(days=1)
    )
    vix_quote = vix_data["Close"].values[0]
    print(f"CBOE VIX quote: {vix_quote[0]}")


def estimate_vix():
    """ Estimates VIX using estimator VIX_t. """

    # Chain calls corresponding to S&P 500 on certain date
    ticker = yf.Ticker(spx_symbol)
    expiry_date = "2025-04-28"  # Fixed to approximate a 30-day horizon as per CBOE
    calls = ticker.option_chain(expiry_date).calls
    puts = ticker.option_chain(expiry_date).puts

    # Compute mid prices for puts and calls
    # This estimates the true market value
    calls["mid_call"] = (calls["bid"] + calls["ask"]) / 2
    puts["mid_put"] = (puts["bid"] + puts["ask"]) / 2
    options = pd.merge(
        calls[["strike", "mid_call"]], puts[["strike", "mid_put"]], on="strike"
    )

    tau = 30 / 365
    r = 0.05  # risk-free rate
    F = calculate_F(calls, puts, r, tau)

    puts_otm = options[options["strike"] < F]
    calls_otm = options[options["strike"] > F]
    put_part = vix_integral(puts_otm, "mid_put")
    call_part = vix_integral(calls_otm, "mid_call")

    # Final calculations formula
    vix_squared = (2 * np.exp(r * tau) / tau) * (put_part + call_part)
    vix_estimate = np.sqrt(vix_squared)
    vix_estimate = 100 * vix_estimate # To convert from decimal to percentage.

    print(f"Estimated VIX: {vix_estimate:.2f}")

spx_symbol = "^SPX"
today_str = "2025-03-28"  # This date needs to stay fixed
end_date = datetime.datetime.strptime(today_str, "%Y-%m-%d").date()
start_date = end_date - datetime.timedelta(days=365)

estimate_vix()
cbeo_vix()
