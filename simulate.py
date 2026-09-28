from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma

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
WEIBULL_SHAPES = (0.75, 1.0, 1.5)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"


def run_exponential_study(samples):
    results = []

    severe_estimates = None
    severe_naive_estimates = None

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

        exact_mcse = np.sqrt(
            exact_coverage * (1 - exact_coverage) / M
        )
        wald_mcse = np.sqrt(
            wald_coverage * (1 - wald_coverage) / M
        )
        log_mcse = np.sqrt(
            log_coverage * (1 - log_coverage) / M
        )

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
            "exact_mcse": exact_mcse,
            "wald_mcse": wald_mcse,
            "log_wald_mcse": log_mcse,
        })

        print(f"r = {r}")
        print(f"MLE mean:       {estimates.mean():.2f}")
        print(f"MLE SD:         {estimates.std(ddof=1):.2f}")
        print(f"Naive mean:     {naive_estimates.mean():.2f}")
        print(f"Exact coverage: {exact_coverage:.4f}")
        print(f"Wald coverage:  {wald_coverage:.4f}")
        print(f"Log coverage:   {log_coverage:.4f}")
        print()

    relative_loss = 100 * (
        1 - severe_naive_estimates.mean() / THETA
    )

    print("Severe censoring (r = 10)")
    print(f"MLE median:          {np.median(severe_estimates):.2f}")
    print(f"Naive median:        {np.median(severe_naive_estimates):.2f}")
    print(f"Naive relative loss: {relative_loss:.2f}%")
    print()

    return (
        pd.DataFrame(results),
        severe_estimates,
        severe_naive_estimates,
    )


def run_weibull_study(base_exponentials):
    results = []

    for k in WEIBULL_SHAPES:
        scale = THETA / gamma(1 + 1 / k)

        weibull_samples = scale * base_exponentials ** (1 / k)
        weibull_samples.sort(axis=1)

        for r in R_VALUES:
            exposures = np.array([
                total_time_on_test(sample, r)
                for sample in weibull_samples
            ])

            estimates = mle(exposures, r)

            exact_lower, exact_upper = exact_confidence_interval(
                exposures, r
            )
            wald_lower, wald_upper = wald_confidence_interval(
                exposures, r
            )

            relative_bias = 100 * (
                estimates.mean() - THETA
            ) / THETA

            exact_coverage = np.mean(
                (exact_lower <= THETA) & (THETA <= exact_upper)
            )
            wald_coverage = np.mean(
                (wald_lower <= THETA) & (THETA <= wald_upper)
            )

            results.append({
                "k": k,
                "r": r,
                "mean_estimate": estimates.mean(),
                "relative_bias_percent": relative_bias,
                "exact_coverage": exact_coverage,
                "wald_coverage": wald_coverage,
            })

    return pd.DataFrame(results)


def plot_severe_censoring(estimates, naive_estimates):
    bins = np.arange(0, 15000 + 125, 125)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(estimates, bins=bins, density=True)
    axes[0].axvline(THETA, linestyle="--")
    axes[0].set_xlim(0, 15000)
    axes[0].set_xlabel("Estimate (hours)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("MLE under Type-II censoring")

    axes[1].hist(naive_estimates, bins=bins, density=True)
    axes[1].axvline(THETA, linestyle="--")
    axes[1].set_xlim(0, 15000)
    axes[1].set_xlabel("Estimate (hours)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Naive estimator")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "severe_censoring_histograms.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_weibull_sensitivity(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for r in R_VALUES:
        subset = results[results["r"] == r]

        axes[0].plot(
            subset["k"],
            subset["relative_bias_percent"],
            marker="o",
            label=f"r = {r}",
        )
        axes[1].plot(
            subset["k"],
            subset["exact_coverage"],
            marker="o",
            label=f"r = {r}",
        )

    axes[0].axhline(0, linestyle="--")
    axes[0].set_xlabel("Weibull shape k")
    axes[0].set_ylabel("Relative bias (%)")
    axes[0].set_title("Bias under model misspecification")
    axes[0].legend()

    axes[1].axhline(0.95, linestyle="--")
    axes[1].set_xlabel("Weibull shape k")
    axes[1].set_ylabel("Coverage")
    axes[1].set_title("Exact exponential interval coverage")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "weibull_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    rng = np.random.Generator(np.random.PCG64(SEED))

    samples = rng.exponential(
        scale=THETA,
        size=(M, N),
    )

    base_exponentials = samples / THETA
    samples.sort(axis=1)

    (
        exponential_results,
        severe_estimates,
        severe_naive_estimates,
    ) = run_exponential_study(samples)

    weibull_results = run_weibull_study(base_exponentials)

    exponential_results.to_csv(
        RESULTS_DIR / "simulation_results.csv",
        index=False,
    )
    weibull_results.to_csv(
        RESULTS_DIR / "weibull_sensitivity.csv",
        index=False,
    )

    plot_severe_censoring(
        severe_estimates,
        severe_naive_estimates,
    )
    plot_weibull_sensitivity(weibull_results)


if __name__ == "__main__":
    main()