"""
Marketing KPIs Module

This module provides functions for computing marketing Key Performance Indicators (KPIs)
using Bayesian statistical methods. It implements pooled Bayesian models for estimating
lead generation and conversion rates, enabling data-driven marketing decisions with
uncertainty quantification.

Key Features:
- Bayesian lead and conversion rate estimation
- Monte Carlo simulation for posterior predictive distributions
- Confidence-based KPI threshold computation
- Support for custom priors and historical data integration

Functions:
- compute_lower_q_proba: Compute quantile thresholds for conversion targets
- estimate_lead_conversions_kpi: Main function for KPI estimation using pooled Bayesian model

Dependencies:
- numpy: Numerical computations
- pandas: Data manipulation and analysis
- scipy.stats: Statistical distributions (beta, betabinom, gamma)

Author: Moses Kabungo
Date: September 2025
"""

import numpy as np
import pandas as pd
from scipy.stats import beta, betabinom, gamma


def compute_lower_q_proba(q: float, n_new, alpha_past, beta_past):
    """
    Compute the quantile threshold
    """

    k_q = int(betabinom.ppf(q, n_new, alpha_past, beta_past))
    return k_q


def estimate_lead_conversions_kpi(
    csv_url: str,
    q: list[float] = None,
    n_sim: int = 100_000,
    priors: dict = None,
    random_state: int = 0,
) -> dict:
    """
    Estimate marketing KPIs using a pooled Bayesian model.

    Args:
        csv_url (str): URL or path to a CSV file with columns:
            - 'Leads' (int, nonnegative)
            - 'Conversions' (int, nonnegative, <= Leads)
            - 'Month' (string, optional, for reference only)
        q (list[float], optional): Quantiles to compute for KPI thresholds.
            Represents the probability that conversions meet or exceed the target.
            Defaults to [0.5, 0.8, 0.95, 0.99] if the parameter was not specified..
        n_sim (int, optional): Number of Monte Carlo simulations. Default = 100,000.
        priors (dict, optional): Hyperparameters for priors.
            Keys:
                - 'alpha_leads', 'beta_leads' : Gamma prior for λ (lead rate)
                - 'alpha_conv', 'beta_conv'   : Beta prior for p (conversion rate)
            Default priors are data-informed for better realism.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        dict: Contains:
            - 'expected_conversions': Expected number of conversions (float)
            - 'kpi_table': DataFrame with confidence levels and conversion targets

    Note:
        The KPI table shows the minimum conversions target T for each confidence level.
        For example, at 95% confidence, P(conversions >= T) = 0.95.
    """
    # --- Reproducible result ---
    rng = np.random.default_rng(random_state)
    q = np.asarray(q)

    # --- Load and Validate Data ---
    data = pd.read_csv(csv_url)

    if priors is None:
        priors = [0.5, 0.8, 0.95, 0.99]

    # Check required columns
    required_cols = {"Leads", "Conversions"}
    if not required_cols.issubset(data.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, got {set(data.columns)}"
        )

    # Validate data values
    if not np.issubdtype(data["Leads"].dtype, np.integer):
        raise ValueError("'Leads' must be integers.")
    if not np.issubdtype(data["Conversions"].dtype, np.integer):
        raise ValueError("'Conversions' must be integers.")

    if (data["Leads"] < 0).any():
        raise ValueError("'Leads' must be non-negative.")
    if (data["Conversions"] < 0).any():
        raise ValueError("'Conversions' must be non-negative.")
    if (data["Conversions"] > data["Leads"]).any():
        raise ValueError("'Conversions' cannot exceed 'Leads' in any row.")

    total_leads = int(data["Leads"].sum())
    total_conversions = int(data["Conversions"].sum())
    n_months = len(data)

    # --- Set Default Priors Based on Data Scale ---
    if priors is None:
        # Estimate average monthly leads and conversion rate for informed defaults
        avg_leads = total_leads / n_months if n_months > 0 else 1.0
        avg_conv_rate = total_conversions / total_leads if total_leads > 0 else 0.01

        # Set priors: Gamma for leads, Beta for conversion rate
        priors = {
            "alpha_leads": 1.0,
            "beta_leads": 1.0 / avg_leads if avg_leads > 0 else 1.0,
            "alpha_conv": 1.0,
            "beta_conv": (
                (1.0 - avg_conv_rate) / avg_conv_rate if avg_conv_rate > 0 else 99.0
            ),
        }

    # --- Posterior Calculations ---
    # For λ (lead rate): Gamma posterior
    alpha_post_lambda = priors["alpha_leads"] + total_leads
    beta_post_lambda = priors["beta_leads"] + n_months

    # For p (conversion rate): Beta posterior
    alpha_post_p = priors["alpha_conv"] + total_conversions
    beta_post_p = priors["beta_conv"] + (total_leads - total_conversions)

    # --- Monte Carlo Simulation ---
    lambda_samples = gamma.rvs(
        a=alpha_post_lambda, scale=1 / beta_post_lambda, size=n_sim, random_state=rng
    )
    p_samples = beta.rvs(a=alpha_post_p, b=beta_post_p, size=n_sim, random_state=rng)

    # Directly simulate conversions using Poisson identity
    conv_pred = rng.poisson(lam=lambda_samples * p_samples)

    # --- Summaries ---
    expected_conversions = np.mean(conv_pred)  # Keep as float for precision
    quantiles = np.quantile(conv_pred, 1 - q)  # 1-q for lower bound

    kpi_table = pd.DataFrame(
        {
            "Confidence": [f"{int(level*100)}%" for level in q],
            "Minimum Conversions Target (T)": quantiles.astype(int),
        }
    )

    return {"expected_conversions": expected_conversions, "kpi_table": kpi_table}
