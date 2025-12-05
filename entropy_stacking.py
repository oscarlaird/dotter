# Given a mixture of gaussians with different, but known, pi_i
# how should we position their mu_i along the unit 1-Torus
# So that the entropy of the mixture is maximized?

# Theory
# Consider an N-dimensional sample as N → ∞
# Let i be the index of the vertex from which the sample truly originated
# Then, H(X) = H(X_o) + I(i; X) where X_o is the entropy when the center is known
# Proof:
# H(i, X) = H(i) + H(X_o)
# H(X) + H(i | X) = H(i) + H(X_o)
# H(X) = H(X_o) + I(i; X)

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
    I = 1/2 * np.log(2) - 1/4 - np.log(Z) - (1/(8 * sigma**2)) * expected_d2
    return I

def x_to_I(x):
    mu = x_to_mu(x)
    rho, d = pi_mu_to_rho_d(pi, mu)
    return compute_I(rho, d, sigma)
# numeric

# grid search for the optimal x
x_values = np.linspace(0, 0.5, 1000)
I_values = [x_to_I(x) for x in x_values]
best_idx = np.argmax(I_values)
best_x = x_values[best_idx]
best_I = I_values[best_idx]
print(f"Best x: {best_x:.4f}")
print(f"Maximum I: {best_I:.6f}")
plt.plot(x_values, I_values), x_to_I(0.25)
#%%
mu_opt = x_to_mu(best_x)
def mixed_gaussian(x):
    prob = 0
    for i in range(len(pi)):
        dist = min(np.abs(x - mu_opt[i]), 1 - np.abs(x - mu_opt[i]))
        prob += pi[i] * stats.norm.pdf(dist, 0, sigma)
    return prob
xs = np.linspace(0, 1, 1000)
plt.plot(xs, [mixed_gaussian(x) for x in xs])