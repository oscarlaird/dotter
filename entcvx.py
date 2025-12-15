#%%

import cvxpy as cp
import numpy as np

def find_post_rho(sep):
    J = 2
    sigma = 1.0
    m = 0.0
    sep = sep
    pi = np.array([0.5, 0.5])
    mu = np.array([0.0, sep])
    rho = cp.Variable(J, nonneg=True, name="rho")
    y = cp.Variable(J, name="y")  # y = ρ x
    # --- 3. Objective Function (Corrected) ---
    # We must iterate over N to create a quad_over_lin term for each element.
    # This calculates: sum_j [ (y_j - rho_j * mu_j)^2 / rho_j ]
    # Note: y[j] is scalar, rho[j] is scalar, so quad_over_lin works here.
    perspective_terms = cp.sum([
        cp.quad_over_lin(y[j] - rho[j] * mu[j], rho[j]) 
        for j in range(J)
    ])
    rate_cost = (1 / 2 * sigma**2) * cp.sum(perspective_terms)
    kl_cost = cp.sum(cp.rel_entr(rho, pi))
    objective = cp.Minimize(rate_cost + kl_cost)

    constraints = [
        cp.sum(rho) == 1,
        # Constraint 2: sum_j rho_j x_j = m  <==> sum_j y_j = m
        cp.sum(y) == m,
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
import matplotlib.pyplot as plt

separations = np.linspace(0, 10, 25)
plt.plot(separations, [entropy_for_separation(x) for x in separations])
plt.plot(separations, [cramer4_H_X(x) for x in separations])
plt.show()