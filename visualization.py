import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist, binom, gamma as gamma_dist, poisson, norm

# ---------------------------
# 1. Beta–Binomial
# ---------------------------
def plot_beta_binomial(alpha, beta, 
                       n=None, k=None, 
                       xlim=(0, 1), 
                       num_points=200, 
                       prior_color="navy", 
                       post_color="crimson", 
                       likelihood_color="darkorange", 
                       linewidth=2, 
                       title=None, 
                       xlabel=r"$\theta$", 
                       ylabel="Density / Scaled PMF", 
                       grid=True, 
                       show=True, 
                       figsize=(7, 5)):
    x = np.linspace(xlim[0], xlim[1], num_points)
    y_prior = beta_dist.pdf(x, alpha, beta)

    plt.figure(figsize=figsize)
    plt.plot(x, y_prior, color=prior_color, linewidth=linewidth, 
             label=fr"Prior: Beta($\alpha={alpha},\beta={beta}$)")

    if n is not None and k is not None:
        likelihood = binom.pmf(k, n, x)
        likelihood_scaled = likelihood / likelihood.max() * y_prior.max()
        plt.plot(x, likelihood_scaled, color=likelihood_color, linestyle="--", 
                 linewidth=linewidth, label=fr"Likelihood: Binomial($n={n},k={k}$)")

        alpha_post = alpha + k
        beta_post = beta + n - k
        y_post = beta_dist.pdf(x, alpha_post, beta_post)
        plt.plot(x, y_post, color=post_color, linewidth=linewidth, 
                 label=fr"Posterior: Beta($\alpha={alpha_post},\beta={beta_post}$)")

    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        title = "Bayesian Update: Beta–Binomial"
    plt.title(title)
    if grid: plt.grid(alpha=0.3)
    plt.legend()
    if show: plt.show()

# ---------------------------
# 2. Gamma–Poisson
# ---------------------------
def plot_gamma_poisson(alpha, beta, 
                       n=None, observed=None, 
                       xlim=(0, 20), 
                       num_points=200, 
                       prior_color="navy", 
                       post_color="crimson", 
                       likelihood_color="darkorange", 
                       linewidth=2, 
                       title=None, 
                       xlabel=r"$\lambda$", 
                       ylabel="Density / Scaled PMF", 
                       grid=True, 
                       show=True, 
                       figsize=(7, 5)):
    x = np.linspace(xlim[0], xlim[1], num_points)
    y_prior = gamma_dist.pdf(x, a=alpha, scale=1/beta)

    plt.figure(figsize=figsize)
    plt.plot(x, y_prior, color=prior_color, linewidth=linewidth, 
             label=fr"Prior: Gamma($\alpha={alpha},\beta={beta}$)")

    if n is not None and observed is not None:
        k_sum = np.sum(observed)
        likelihood = poisson.pmf(k_sum, mu=x*n)
        likelihood_scaled = likelihood / likelihood.max() * y_prior.max()
        plt.plot(x, likelihood_scaled, color=likelihood_color, linestyle="--", 
                 linewidth=linewidth, label=fr"Likelihood: Poisson(sum={k_sum}, n={n})")

        alpha_post = alpha + k_sum
        beta_post = beta + n
        y_post = gamma_dist.pdf(x, a=alpha_post, scale=1/beta_post)
        plt.plot(x, y_post, color=post_color, linewidth=linewidth, 
                 label=fr"Posterior: Gamma($\alpha={alpha_post},\beta={beta_post}$)")

    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        title = "Bayesian Update: Gamma–Poisson"
    plt.title(title)
    if grid: plt.grid(alpha=0.3)
    plt.legend()
    if show: plt.show()

# ---------------------------
# 3. Normal–Normal (known variance)
# ---------------------------
def plot_normal_normal(mu0, tau2, 
                       data=None, sigma2=1, 
                       xlim=(-5, 5), 
                       num_points=200, 
                       prior_color="navy", 
                       post_color="crimson", 
                       likelihood_color="darkorange", 
                       linewidth=2, 
                       title=None, 
                       xlabel=r"$\mu$", 
                       ylabel="Density", 
                       grid=True, 
                       show=True, 
                       figsize=(7, 5)):
    x = np.linspace(xlim[0], xlim[1], num_points)
    y_prior = norm.pdf(x, mu0, np.sqrt(tau2))

    plt.figure(figsize=figsize)
    plt.plot(x, y_prior, color=prior_color, linewidth=linewidth, 
             label=fr"Prior: Normal($\mu_0={mu0}, \tau^2={tau2}$)")

    if data is not None:
        n = len(data)
        xbar = np.mean(data)

        post_var = 1 / (1/tau2 + n/sigma2)
        post_mean = post_var * (mu0/tau2 + n*xbar/sigma2)

        likelihood = norm.pdf(x, xbar, np.sqrt(sigma2/n))
        likelihood_scaled = likelihood / likelihood.max() * y_prior.max()
        plt.plot(x, likelihood_scaled, color=likelihood_color, linestyle="--", 
                 linewidth=linewidth, label=fr"Likelihood: $\bar{{x}}={xbar:.2f}, n={n}$")

        y_post = norm.pdf(x, post_mean, np.sqrt(post_var))
        plt.plot(x, y_post, color=post_color, linewidth=linewidth, 
                 label=fr"Posterior: Normal($\mu={post_mean:.2f}, \tau^2={post_var:.2f}$)")

    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is None:
        title = "Bayesian Update: Normal–Normal"
    plt.title(title)
    if grid: plt.grid(alpha=0.3)
    plt.legend()
    if show: plt.show()
