# %%
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob
import numpy as np

pi0 = 0.9
FWHM = 4
dim = 25
alpha = 0.1

if __name__ == "__main__":
    pi0 = sys.argv[1]
    FWHM = sys.argv[2]

files = glob.glob(f"../figures/results_pi0{pi0}_fwhm{FWHM}_ntest*_dim{dim}.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
N = df["repeats"].iloc[0] 
assert df["repeats"].nunique() == 1, "several values of repeats in one CSV"
assert (df["pi0"] == pi0).all(), f"pi0 mismatch in loaded files"
assert (df["FWHM"] == FWHM).all(), f"FWHM mismatch in loaded files"
df = df[df["n_test"] >= 20]

summary = (
    df.groupby(["method", "n_test"])[["jer", "power"]]
    .agg(["mean", "std"])
)
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()
summary["jer_se"] = 2 * np.sqrt(summary["jer_mean"] * (1 - summary["jer_mean"]) / N)

# %% JER plot
plt.figure(figsize=(6, 4))
for method, g in summary.groupby("method"):
    g = g.sort_values("n_test")
    plt.plot(g["n_test"], g["jer_mean"], label=method)
    plt.fill_between(
        g["n_test"], g["jer_mean"] - g["jer_se"], g["jer_mean"] + g["jer_se"], alpha=0.2
    )

plt.xscale("log")
plt.axhline(y=alpha, color="red", linestyle="--", label=f"α = {alpha}")
plt.xlabel("n")
plt.ylabel("Joint Error Rate (JER)")
plt.title(rf"$\pi_0 = {pi0}$, FWHM = {FWHM}")
plt.legend(prop={"size": 8})
plt.savefig(f"../figures/fig_jer_n_tests_pi0{pi0}_fwhm{FWHM}.pdf", bbox_inches="tight")

# %% Power plot
plt.figure(figsize=(6, 4))
for method, g in summary.groupby("method"):
    g = g.sort_values("n_test")
    plt.plot(g["n_test"], g["power_mean"], label=method)

plt.xscale("log")
plt.xlabel("n")
plt.ylabel("Power")
plt.title(rf"$\pi_0 = {pi0}$, FWHM = {FWHM}")
plt.legend(prop={"size": 8})
plt.savefig(f"../figures/fig_power_n_tests_pi0{pi0}_fwhm{FWHM}.pdf", bbox_inches="tight")
