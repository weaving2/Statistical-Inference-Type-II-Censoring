import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.inference import (
    total_time_on_test,
    mle,
    naive_estimator,
    exact_confidence_interval,
    wald_confidence_interval,
    log_wald_confidence_interval,
)


THETA = 5000
N = 50
R_VALUES = (50, 40, 10)
M = 10_000
SEED = 20260926


rng = np.random.Generator(np.random.PCG64(SEED))

samples = rng.exponential(
    scale=THETA,
    size=(M, N),
)

samples.sort(axis=1)

results = []

for r in R_VALUES:
    exposures = []
    estimates = []
    naive_estimates = []

    for sample in samples:
        exposure = total_time_on_test(sample, r)

        exposures.append(exposure)
        estimates.append(mle(exposure, r))
        naive_estimates.append(naive_estimator(sample, r))

    exposures = np.array(exposures)
    estimates = np.array(estimates)
    naive_estimates = np.array(naive_estimates)

    if r == 10:
        severe_estimates = estimates.copy()
        severe_naive_estimates = naive_estimates.copy()

    exact_lower, exact_upper = exact_confidence_interval(exposures, r)
    wald_lower, wald_upper = wald_confidence_interval(exposures, r)
    log_lower, log_upper = log_wald_confidence_interval(exposures, r)

    exact_coverage = np.mean(
        (exact_lower <= THETA) & (THETA <= exact_upper)
    )
    wald_coverage = np.mean(
        (wald_lower <= THETA) & (THETA <= wald_upper)
    )
    log_coverage = np.mean(
        (log_lower <= THETA) & (THETA <= log_upper)
    )
    exact_length = np.mean(exact_upper - exact_lower)
    wald_length = np.mean(wald_upper - wald_lower)
    log_length = np.mean(log_upper - log_lower)

    exact_mcse = np.sqrt(exact_coverage * (1 - exact_coverage) / M)
    wald_mcse = np.sqrt(wald_coverage * (1 - wald_coverage) / M)
    log_mcse = np.sqrt(log_coverage * (1 - log_coverage) / M)
    
    results.append({
    "r": r,
    "mle_mean": estimates.mean(),
    "mle_sd": estimates.std(ddof=1),
    "naive_mean": naive_estimates.mean(),
    "exact_coverage": exact_coverage,
    "wald_coverage": wald_coverage,
    "log_wald_coverage": log_coverage,
    "exact_length": exact_length,
    "wald_length": wald_length,
    "log_wald_length": log_length,
    })
    
    print(f"r = {r}")
    print(f"MLE mean:      {estimates.mean():.2f}")
    print(f"MLE SD:        {estimates.std(ddof=1):.2f}")
    print(f"Naive mean:    {naive_estimates.mean():.2f}")
    print(f"Exact coverage: {exact_coverage:.4f}")
    print(f"Wald coverage:  {wald_coverage:.4f}")
    print(f"Log coverage:   {log_coverage:.4f}")
    print(f"Exact length:   {exact_length:.2f}")
    print(f"Wald length:    {wald_length:.2f}")
    print(f"Log length:     {log_length:.2f}")
    print()

relative_loss = 100 * (
    1 - severe_naive_estimates.mean() / THETA
)

print(f"Naive relative loss: {relative_loss:.2f}%")

results_df = pd.DataFrame(results)
results_df.to_csv("results/simulation_results.csv", index=False)

bins = np.arange(0, 15000 + 125, 125)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(severe_estimates, bins=bins, density=True)
axes[0].axvline(THETA, linestyle="--")
axes[0].set_xlim(0, 15000)
axes[0].set_xlabel("Estimate (hours)")
axes[0].set_ylabel("Density")
axes[0].set_title("MLE under Type-II censoring")

axes[1].hist(severe_naive_estimates, bins=bins, density=True)
axes[1].axvline(THETA, linestyle="--")
axes[1].set_xlim(0, 15000)
axes[1].set_xlabel("Estimate (hours)")
axes[1].set_ylabel("Density")
axes[1].set_title("Naive estimator")

fig.tight_layout()

fig.savefig(
    "results/figures/severe_censoring_histograms.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)