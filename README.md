# Statistical Inference under Type-II Censoring

Statistical analysis of exponential lifetime data under Type-II censoring.

This project studies maximum likelihood estimation, exact and asymptotic
confidence intervals, and their finite-sample behavior through Monte Carlo
simulation.

## Problem

Suppose \(n\) components are tested simultaneously, but the experiment stops
after the \(r\)-th failure. We observe the first \(r\) failure times while the
remaining \(n-r\) components are censored.

The project estimates the exponential mean lifetime and compares exact and
asymptotic inference under different censoring levels.

## Methods

The project includes:

- Maximum likelihood estimation under Type-II censoring
- Exact confidence intervals based on the chi-square distribution
- Wald and log-Wald confidence intervals
- Monte Carlo simulation with 10,000 replications
- Comparison with the naive estimator that ignores censored observations
- Empirical coverage and interval-length analysis

## Project Structure

```text
.
├── docs/
├── results/
│   └── figures/
├── src/
│   ├── __init__.py
│   └── inference.py
├── tests/
│   └── test_inference.py
├── simulate.py
├── requirements.txt
└── README.md