# Given a mixture of gaussians with different, but known, pi_i
# how should we position their mu_i along the unit 1-Torus
# So that the entropy of the mixture is maximized?

# Theory
# Consider an N-dimensional sample as N → ∞
# Let i be the index of the vertex from which the sample truly originated
# Then, H(X) = H(X_o) + I(i; X) where X_o is the entropy when the center is known
# Proof:
# H(X) = H(X_o, i) - H(i | X)
# = H(X_o) + H(i) - H(i | X)
# = H(X_o) + I(i; X)

# But since σ is known and fixed, so is H(X_o)
# and the goal is to maximize I(i; X)
# so we want X to carry a lot of information about i
# We notice that given our use of (possibly truncated) gaussian noise
# The likelihood of i is captured by observing the distance between the sample and the vertex

# The prior distribution of distance to the vertices is
# D = Σ D_i where D_i = (a_i + ε - b_i)**2
# where a_i is the position of the sample's origin, and b_i is the position of the vertex
#
# Then P(D_bar < x) =~ exp( - N I(x)) by Cramér's theorem
# and
# P(X | D_bar=x) ∝ exp( - N x / 2σ^2) is the likelihood for (truncated) gaussian noise
# So the log-posterior distribution of D_bar is (up to a constant)
# log P(D_bar=x | X) = - N x / 2σ^2 - N I(x)
# Setting the derivative to 0 yields
# I'(x) + 1 / (2σ^2) = 0
# So I'(x) = - 1 / (2σ^2)
# The true origin of the sample (i) will have a distance similar to the mode of the posterior distribution of D
# so we have learned N I(x) bits about the origin of the sample
# The info gain per use of the channel is thus I(x)

# Now it is possible to get a closed form for the rate function as a function of theta i.e. I'(x)
# So we only need evaluate I(x) for θ=-1/2σ^2 to determine the info gain

# we note that a_i and b_i are iid discrete random variables
# and ε is normally distributed
# After a lot of algebra (for which I used gemini)
# we get the following expression for I when I'(x) = -1/2σ^2
#
# Let Z = sum_k ρ_k exp( - d_k**2 / 4σ^2)
# I = 1/2 * ln(2) - 1/4
#     - ln[Z]
#     - (1/(8σ^2)) * (1/Z) * sum_k (ρ_k * d_k**2 * exp( - d_k**2 / 4σ^2))
#
# We can recognize the final term as something like
# the expected value of d_k**2 subject to a boltzmann
# distribution over the possible distances

# Example
# Let's try to optimize the positions when
# K=3
# sigma=0.100
# pi=.5, .25, .25
# mu=0, x, 1-x

#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
x = torch.Tensor()
pi = np.array([.70, .1, .1, .1])
sigma = 0.050

def x_to_mu(x):
    return np.array([0, x, 0.5, 1-x])

def pi_mu_to_rho_d(pi, mu):
    rho = pi[:,None] * pi[None,:]
    rho = rho.flatten()
    d = np.abs(mu[:,None] - mu[None,:])
    d = np.minimum(d, 1 - d)  # use the wrapped distance
    d = d.flatten()
    return rho, d

def compute_I(rho, d, sigma):
    d2 = d**2
    Z = np.sum(rho * np.exp( - d2 / (4 * sigma**2)))
    expected_d2 = np.sum(rho * d2 * np.exp( - d2 / (4 * sigma**2))) / Z
    # I = 1/2 * np.log(2) - 1/4 - np.log(Z) - (1/(8 * sigma**2)) * expected_d2
    # ignore the constant for now
    I = - np.log(Z) - (1/(8 * sigma**2)) * expected_d2
    return I

def x_to_I(x):
    mu = x_to_mu(x)
    rho, d = pi_mu_to_rho_d(pi, mu)
    return compute_I(rho, d, sigma)

#%%
# Numerically evaluate the quality of the approximation on R^1
means = np.array([0, 3])
variances = np.array([1, 1])
weights = np.array([0.5, 0.5])
# Mutual = H(X) - H(E)
def gmm_pdf(x, means, sigma, weights):
    prob = 0
    for i in range(len(means)):
        prob += weights[i] * stats.norm.pdf(x, means[i], sigma)
    return prob
def empirical_entropy(xs, vals):
    # Estimate the entropy using the trapezoidal method
    densities = vals
    # Avoid log(0); where density is zero, set to a small positive value
    densities_safe = np.where(densities > 0, densities, 1e-300)
    integrand = densities_safe * np.log(densities_safe)
    entropy = -np.trapz(integrand, xs)
    return entropy
def entropy_for_separation(separation):
    means = np.array([0, separation])
    sigma = 1
    weights = np.array([0.5, 0.5])
    xs = np.linspace(-10*sigma, separation + 10*sigma, 200)
    vals = np.array([gmm_pdf(x, means, sigma, weights) for x in xs])
    return empirical_entropy(xs, vals)
def entropy_for_sigma(sigma):
    means = np.array([0, 1])
    weights = np.array([0.5, 0.5])
    xs = np.linspace(-10*sigma, 1 + 10*sigma, 200)
    vals = np.array([gmm_pdf(x, means, sigma, weights) for x in xs])
    return empirical_entropy(xs, vals)
def H_normal(sigma):
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)
separations = np.linspace(0, 10, 100)
plt.plot(separations, [entropy_for_separation(x) for x in separations])
# plot bounds
plt.hlines([H_normal(1), H_normal(1)+np.log(2)], xmin=0, xmax=10, color='red', linewidth=.5)

# cramer
def cramer_entropy_for_separation(separation):
    # H(X) = Mutual + H(E)
    means = np.array([0, separation])
    rho, d = pi_mu_to_rho_d(weights, means)
    return compute_I(rho, d, 1) + H_normal(1)
plt.plot(separations, [cramer_entropy_for_separation(x) for x in separations] - (cramer_entropy_for_separation(0) - entropy_for_separation(0)))


#%%
# cramer2
# H(X) = H(A) + H(E) - H(A|X)
def binary_entropy(p):
    return -p * np.log(p) - (1 - p) * np.log(1 - p)
def deriv_binary_entropy(p):
    return np.log(1-p) - np.log(p)
def A2_post_mode(s):
    print(f"s: {s}")
    # What is the mode of ||A||_2^2 after observing ||X||_2^2 = s?
    # xs = np.linspace(0, 1, 3000)
    # d_ll = (s - xs)/s**2
    # d_prior = np.log(1 - xs) - np.log(xs)
    # np.bisect does not exist; use scipy.optimize.bisect instead
    from scipy.optimize import bisect
    root = bisect(lambda x: (s - x) / s**2 + np.log(1 - x) - np.log(x), 1e-6, 1 - 1e-6)
    return root
def H_A_given_X_LB(s):
    root = A2_post_mode(s)
    return binary_entropy(root)
def cramer2_H_X(s):
    return np.log(2) + H_normal(1) - H_A_given_X_LB(s)

separations = np.linspace(0.1, 10, 100)
plt.plot(separations, [cramer2_H_X(x) for x in separations])
plt.plot(separations, [entropy_for_separation(x) for x in separations])

#%%
# cramer3
from functools import partial
from scipy.optimize import bisect
def x_from_h(h, s2, m):
    beta = 1 - 2*h*s2
    return s2 / beta + m**2 / beta**2
x1_from_h = partial(x_from_h, m=1)
x2_from_h = partial(x_from_h, m=0)
def rho_from_h(h, s2):
    return (s2 - x2_from_h(h, s2)) / (x1_from_h(h, s2) - x2_from_h(h, s2))
def d_post_from_h(h, s2):
    rho = rho_from_h(h, s2)
    if rho > 0.99:
        return -np.inf
    d_prior = deriv_binary_entropy(rho)
    d_likelihood = -h * (x1_from_h(h, s2) - x2_from_h(h, s2))
    # !! TODO: this should be d_prior + d_likelihood,
    # but this is necessary bc perhaps a sign error in d_likelihood
    d_total = d_prior - d_likelihood
    return d_total
def rho_post_mode(s2):
    print(f"s2: {s2}")
    h_opt = bisect(lambda h: d_post_from_h(h, s2), -10, -1e-6)
    return rho_from_h(h_opt, s2)
def cramer3_H_A_given_X(s2):
    rho = rho_post_mode(s2)
    return binary_entropy(rho)
def cramer3_H_X(sigma):
    s2 = sigma**2
    return np.log(2) + H_normal(1) - cramer3_H_A_given_X(s2)
separations = np.linspace(0.1, 10, 100)
plt.plot(separations, [entropy_for_separation(x) for x in separations])
# plt.plot(sigmas, [rho_post_mode(x**2) for x in sigmas])
plt.plot(separations, [cramer3_H_X(1 / x) for x in separations])
#%%
sigma = 0.7
hs = np.linspace(-1, -0.01, 100)
plt.plot(hs, [d_post_from_h(h, sigma**2) for h in hs])
plt.plot(hs, [rho_from_h(h, sigma**2) for h in hs])
# x1, x2
plt.plot(hs, [x1_from_h(h, sigma**2) for h in hs])
plt.plot(hs, [x2_from_h(h, sigma**2) for h in hs])
#
plt.hlines([0.5], xmin=-1, xmax=-0.01, color='red', linewidth=0.8)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.ylim(-1,2)