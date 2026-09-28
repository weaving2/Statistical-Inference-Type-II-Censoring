import numpy as np
from scipy.stats import chi2, norm


def total_time_on_test(times, r):
    times = np.sort(times)
    n = len(times)

    return np.sum(times[:r]) + (n - r) * times[r - 1]


def mle(exposure, r):
    return exposure / r


def naive_estimator(times, r):
    times = np.sort(times)

    return np.mean(times[:r])


def exact_confidence_interval(exposure, r, alpha=0.05):
    df = 2 * r

    lower = 2 * exposure / chi2.ppf(1 - alpha / 2, df)
    upper = 2 * exposure / chi2.ppf(alpha / 2, df)

    return lower, upper


def wald_confidence_interval(exposure, r, alpha=0.05):
    theta_hat = mle(exposure, r)
    z = norm.ppf(1 - alpha / 2)
    se = theta_hat / np.sqrt(r)

    lower = theta_hat - z * se
    upper = theta_hat + z * se

    return lower, upper


def log_wald_confidence_interval(exposure, r, alpha=0.05):
    theta_hat = mle(exposure, r)
    z = norm.ppf(1 - alpha / 2)

    lower = theta_hat * np.exp(-z / np.sqrt(r))
    upper = theta_hat * np.exp(z / np.sqrt(r))

    return lower, upper