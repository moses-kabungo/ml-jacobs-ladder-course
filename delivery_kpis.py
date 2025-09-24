"""
This module implements Bayesian updating for delivery time estimation
and computation of Service Level Agreements (SLAs) in a logistics setting.
"""

import math
import scipy.stats as st
from typing import Dict, Tuple


def compute_delivery_time_sla(
    prior_mean: float,
    prior_var: float,
    data_var: float,
    sample_size: int,
    sample_mean: float,
    confidence: float = 0.999,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute Bayesian posterior and posterior predictive summaries for delivery time.

    Assumptions
    -----------
    - Prior: θ ~ Normal(prior_mean, prior_var)
    - Data: ȳ ~ Normal(θ, data_var / sample_size)
    - Posterior: θ | y ~ Normal(posterior_mean, posterior_var)
    - Predictive: ỹ ~ Normal(posterior_mean, data_var + posterior_var)

    Parameters
    ----------
    prior_mean : float
        Prior mean of delivery time (μ₀).
    prior_var : float
        Prior variance of delivery time (τ₀²).
    data_var : float
        Known variance of individual delivery times from historical records (σ²).
    sample_size : int
        Number of observed deliveries in the current sample (n).
    sample_mean : float
        Sample mean delivery time from current data (ȳ).
    confidence : float, optional
        Desired one-sided confidence level for SLA cutoff (default=0.999).
    ci_level : float, optional
        Credible interval probability mass for the posterior predictive (default=0.95).

    Returns
    -------
    dict
        Dictionary with:
        - "posterior_mean": float, posterior mean of θ
        - "posterior_var": float, posterior variance of θ
        - "sla_cutoff": float, SLA cutoff time at given confidence
        - "predictive_ci": (low, high), central credible interval for predictive distribution
    """
    # Posterior variance of θ
    posterior_var = 1 / (1 / prior_var + sample_size / data_var)

    # Posterior mean of θ
    posterior_mean = posterior_var * (
        prior_mean / prior_var + (sample_size * sample_mean) / data_var
    )

    # Posterior predictive variance (future observation)
    predictive_var = data_var + posterior_var
    predictive_std = math.sqrt(predictive_var)

    # SLA cutoff (upper quantile)
    z_sla = st.norm.ppf(confidence)
    sla_cutoff = posterior_mean + z_sla * predictive_std

    # Credible interval (central interval for predictive distribution)
    alpha = 1 - confidence
    z_low = st.norm.ppf(alpha / 2)
    z_high = st.norm.ppf(1 - alpha / 2)
    ci = (
        posterior_mean + z_low * predictive_std,
        posterior_mean + z_high * predictive_std,
    )

    return {
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
        "sla_cutoff": sla_cutoff,
        "predictive_ci": ci,
    }
