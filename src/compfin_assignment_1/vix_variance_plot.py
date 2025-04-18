import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

from vix import calculate_F, calc_delta_K, vix_integral, cbeo_vix
from vix import estimate_vix
#from volatility import classical_drift, classical_volatility, parkinson_volatility, rolling_window_comparison


from vol_refac import VolatilityCalculations
from vix import calculate_F, calc_delta_K, vix_integral, cbeo_vix   

calls = pd.read_csv("csv_data/Call_option_data_2025-04-03_final.csv")
puts = pd.read_csv("csv_data/Put_option_data_2025-04-03_final.csv")

calls["mid_call"] = (calls["bid"] + calls["ask"]) / 2
puts["mid_put"] = (puts["bid"] + puts["ask"]) / 2

today = datetime.date(2025, 3,5)
expiry_date = datetime.date(2025, 4, 3)
tau = (expiry_date - today).days / 365
r = 0.02 # risk-free rate

opts = pd.merge(
    calls[["strike", "mid_call"]], puts[["strike", "mid_put"]], on="strike"
)

F = calculate_F(calls, puts, r, tau)
puts_otm = opts[opts["strike"] < F]
calls_otm = opts[opts["strike"] > F]
put_part = vix_integral(puts_otm, "mid_put")
call_part = vix_integral(calls_otm, "mid_call")

vix_2 = (2 * np.exp(r * tau) / tau) * (put_part + call_part)
vix_estimate = 100 * np.sqrt(vix_2)  # convert to percentage

print(f"VIX estimate: {vix_estimate:.2f}")

# vix_q = yf.download("^VIX", start=expiry_date, 
#                     end=expiry_date + datetime.timedelta(days=1),
#                     progress=False)["Close"].iloc[0]

# print(f"VIX quote: {vix_q:.2f}%")


vix_q = yf.download("^VIX", start=today, 
                    end=today + datetime.timedelta(days=1),
                    progress=False)["Close"].iloc[0]
vix_q = float(vix_q)
print(f"VIX quote on {today}: {vix_q:.2f}")

# part d)

hist_start = datetime.datetime(2024, 3, 5)
hist_end = datetime.datetime(2025, 3, 5)

window_length = 30

realized_vol_array, parkinson_vol_array, _ = VolatilityCalculations.compute_vol_rolling_window(
    ticker="^SPX",
    startdate=hist_start,
    enddate=hist_end,
    window_length=window_length,
    #vol_func=classical_volatility,
    #log_returns=True,
    return_implied = False
)


N_wind = len(realized_vol_array)
L = window_length
scale_factor = np.sqrt((N_wind-1) / (L-1)) * 100

real_vol_corrected = realized_vol_array * scale_factor
parkinson_vol_corrected = parkinson_vol_array * scale_factor

vc = VolatilityCalculations("^SPX", hist_start, hist_end)

dates = vc.market_data.index[window_length:]
#dates = pd.date_range(start=hist_start, end=hist_end, freq="B")[window_len-1:]

assert len(dates) == len(realized_vol_array), "Dates and realized vol array lengths do not match."

real_vol_s = pd.Series(real_vol_corrected, index=dates)
real_vol_s = real_vol_s.dropna()
real_vol_annual = real_vol_s * np.sqrt(252)

parkinson_vol_s = pd.Series(parkinson_vol_corrected, index=dates)
parkinson_vol_s = parkinson_vol_s.dropna()
parkinson_vol_annual = parkinson_vol_s #* np.sqrt(252)

vix_df = yf.download(
    "^VIX", start=dates[0], end=dates[-1], progress=False
)

vix_ts = vix_df["Close"].squeeze()

assert isinstance(vix_ts, pd.Series), "VIX time series is not a pandas Series."
assert vix_ts.ndim==1, "VIX time series is not 1-dimensional."

df = pd.DataFrame(
    {
        "Realized Volatility": real_vol_annual,
        "Parkinson Volatility": parkinson_vol_annual,
        "CBOE VIX": vix_ts,
    }
)
df = df.dropna()

corr_vol_cboe = df["Realized Volatility"].corr(df["CBOE VIX"])
print(f"Pearson correlation between Realized Volatility and CBOE VIX: {corr_vol_cboe:.2f}")

# run cointegration test

coint_result = coint(df["Realized Volatility"], df["CBOE VIX"])
print(f"Cointegration test statistic: {coint_result[0]:.2f}")

# run 
corr_park_cboe = df["Parkinson Volatility"].corr(df["CBOE VIX"])
print(f"Pearson correlation between Parkinson Volatility and CBOE VIX: {corr_park_cboe:.2f}")   

#run cointegration test
coint_result_park = coint(df["Parkinson Volatility"], df["CBOE VIX"])
print(f"Cointegration test statistic (Parkinson): {coint_result_park[0]:.2f}")


plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Realized Volatility"], label="Realized Volatility (annualized)", color="blue")
plt.plot(df.index, df["Parkinson Volatility"], label="Parkinson Volatility (annualized)", color="green")
plt.plot(vix_ts.index, vix_ts, label="CBOE VIX", color="orange")
plt.legend()
plt.title("Realized Volatility vs CBOE VIX")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.show()

# e) regressions

spx = yf.download("^SPX",
                  start = df.index[0],
                  end   = df.index[-1] + datetime.timedelta(days=1),
                  progress=False)["Close"]
rets0 = spx.pct_change().dropna()

common = df.index.intersection(rets0.index)
rets = rets0.loc[common]
df = df.loc[common]

# regress on VIX
X1 = sm.add_constant(df["CBOE VIX"])
res1 = sm.OLS(rets, X1).fit()
print(res1.summary())

# regress returns on realized vol
X2 = sm.add_constant(df["Realized Volatility"])
res2 = sm.OLS(rets, X2).fit()
print(res2.summary())