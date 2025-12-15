#%%

import cvxpy as cp
import numpy as np

def find_post_rho(sep):
    J = 2
    sigma = 1.0
    m = 1
    sep = sep
    pi = np.array([0.5, 0.5])
    mu = np.array([0.0, sep])
    rho = cp.Variable(J, nonneg=True, name="rho")
    y = cp.Variable(J, name="y")  # y = ρ x
    # --- 3. Objective Function (Corrected) ---
    # We must iterate over N to create a quad_over_lin term for each element.
    # This calculates: sum_j [ (y_j - rho_j * mu_j)^2 / rho_j ]
    # Note: y[j] is scalar, rho[j] is scalar, so quad_over_lin works here.
    moment = 2
    rate_cost = None
    moment_constraint = None
    if moment == 1:
        perspective_terms = cp.sum([
            cp.quad_over_lin(y[j] - rho[j] * mu[j], rho[j]) 
            for j in range(J)
        ])
        rate_cost = (1 / (2 * sigma**2)) * cp.sum(perspective_terms)
        #
        moment_constraint = cp.sum(y) == m
    elif moment == 2:
        # Rate for X^2
        # I(x) = 1/2 [x/sigma^2 - 1 - log(x/sigma^2)]
        # ρI(x) = ρI(y/ρ)
        #       = 1/2 [ y/σ^2 - ρ + ρ log(ρ/y) + log(sigma^2)]
        perspective_terms = cp.sum([
            1/2 * (y[j]/sigma**2 - rho[j])
            # + 1/2 * cp.log(sigma**2)  # constant, so ignored
            + 1/2 * cp.rel_entr(rho[j], y[j])
            for j in range(J)
        ])
        rate_cost = cp.sum(perspective_terms)
        # 
        moment_constraint = cp.sum([rho[j]*mu[j]**2 + y[j] for j in range(J)]) == m
    else:
        raise ValueError(f"Moment {moment} not supported. Must be 1 or 2.")
    kl_cost = cp.sum(cp.rel_entr(rho, pi))
    objective = cp.Minimize(rate_cost + kl_cost)

    constraints = [
        cp.sum(rho) == 1,
        # Constraint 2: sum_j rho_j x_j = m  <==> sum_j y_j = m
        moment_constraint,
    ]

    problem = cp.Problem(objective, constraints)

    # --- 5. Solve the Problem ---
    problem = cp.Problem(objective, constraints)
    print(f"Problem status: {problem.status}")
    print(f"Problem is DCP: {problem.is_dcp()}")

    # The problem is convex (DCP) and can be solved
    try:
        problem.solve()
        print(f"\nOptimal value: {problem.value:.4f}")

        # --- 6. Recover Optimal x_j and I_j ---
        # Recover x_j from the optimal rho_j and y_j
        # x_j = y_j / rho_j
        x_optimal = y.value / rho.value

        # Calculate the optimal rate functions I_j
        I_optimal = (x_optimal - mu)**2 / (2 * sigma**2)
        
        print("\n--- Optimal Solution ---")
        print(f"Optimal rho_j (weights):\n{rho.value.round(4)}")
        print(f"Optimal x_j (variables):\n{x_optimal.round(4)}")
        print(f"Optimal I_j (rate functions):\n{I_optimal.round(4)}")
        
        # Verification of constraints and objective
        print("\n--- Verification ---")
        print(f"Sum of rho_j: {np.sum(rho.value).round(4)}")
        print(f"Sum of y_j (rho_j * x_j): {np.sum(y.value).round(4)}")
        print(f"Weighted objective sum: {np.sum(rho.value * I_optimal).round(4)}")


    except Exception as e:
        print(f"An error occurred during solving: {e}")

    return rho.value

def H_normal(sigma):
    return 0.5 * np.log(2 * np.pi * np.e) + np.log(sigma)

def shannon_entropy(pi):
    return - np.sum(pi * np.log(pi))

def cramer4_H_X(sep):
    prior = np.array([0.5, 0.5])
    prior_ent = shannon_entropy(prior)
    rho = find_post_rho(sep)
    post_ent = shannon_entropy(rho)
    return prior_ent + H_normal(1) - post_ent

#%%
from scipy import stats
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
def kolchinsky_ent(sep, dist_func, sigma=1.0, means=np.array([0, 1]), weights=np.array([0.5, 0.5])):
    # Kolchinsky & Tracey, Estimating Mixture Entropy with Pairwise Distances
    # H_D(X) := H(X|C) - Sum_i c_i ln( Sum_j c_j exp( - dist_func(p_i||p_j)
    means = np.array([0, sep])
    H_X_given_C = H_normal(sigma)
    t2 = 0
    J = len(means)
    dists = [[dist_func(means[i], means[j], sigma) for j in range(J)] for i in range(J)]
    for i in range(J):
        t = 0
        for j in range(J):
            t += weights[j] * np.exp( - dists[i][j])
        t2 += weights[i] * np.log(t)
    return H_X_given_C - t2
def bhattacharyya_dist(mu1, mu2, sigma):
    return (mu1 - mu2)**2 / (8 * sigma**2)


import matplotlib.pyplot as plt

separations = np.linspace(0, 8, 60)
plt.plot(separations, [entropy_for_separation(x) for x in separations], label='Empirical')
plt.plot(separations, [cramer4_H_X(x) for x in separations], label='Cramer4')
plt.plot(separations, [kolchinsky_ent(x, bhattacharyya_dist) for x in separations], label='Kolchinsky')
plt.legend()
plt.show()