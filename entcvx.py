#%%

import cvxpy as cp
import numpy as np
from scipy import stats

def find_post_rho(sep, moment=2, use_12_m1_constraint=True, gaussian_info_bound=False):
    J = 2
    sigma = 1.0
    sep = sep
    pi = np.array([0.5, 0.5])
    mu = np.array([0.0, sep])
    rho = cp.Variable(J, nonneg=True, name="rho")
    y = cp.Variable(J, name="y")  # y = ρ x
    # --- 3. Objective Function (Corrected) ---
    # We must iterate over N to create a quad_over_lin term for each element.
    # This calculates: sum_j [ (y_j - rho_j * mu_j)^2 / rho_j ]
    # Note: y[j] is scalar, rho[j] is scalar, so quad_over_lin works here.
    rate_cost = None
    moment_constraint = None
    constraints = [
        cp.sum(rho) == 1,
    ]
    if moment == 1:
        m = 0
        perspective_terms = cp.sum([
            cp.quad_over_lin(y[j], rho[j]) 
            for j in range(J)
        ])
        rate_cost = (1 / (2 * sigma**2)) * cp.sum(perspective_terms)
        #
        moment_constraint = cp.sum(y) + cp.sum(cp.multiply(rho, mu)) == m
        constraints.append(moment_constraint)
    elif moment == 2:
        m = sigma**2
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
        # TODO!: Is this wrong?! We don't know that E[X]=0 when we put this constraint on E[X^2]
        # But I tend to think that this is the most likely (in fact, easy to prove from cvxity)
        moment_constraint = cp.sum([rho[j]*mu[j]**2 + y[j] for j in range(J)]) == m
        constraints.append(moment_constraint)
    elif moment == 12:
        m1 = 0.0
        m2 = sigma**2
        # Observe both the first and second moments.
        # x_1 = sum x_i; x_2 = sum x_i^2;
        # I(x_1, x_2) = 1/2 [x_2/sigma^2 - 1 - log(x_2 - x_1^2) + log(sigma^2)]
        # we use the perspective transformation y_dj = x_dj * rho_j
        # The tricky term is
        # f(rho, y1, y2) = -rho log((y2/rho) - (y1/rho)^2)
        # using an auxiliary variable tau_j >= ...
        # y2/rho >= (y1/rho)^2 + exp(-tau/rho)
        # y2 >= y1^2/rho + rho * exp(-tau/rho)
        # which terms are quad_over_lin and ExpCone
        y1 = cp.Variable(J, name="y1")
        y2 = cp.Variable(J, name="y2")
        tau = cp.Variable(J, name="tau")
        z_exp = cp.Variable(J, name="z_exp")
        # auxiliary constraints
        constraints.extend([
            cp.ExpCone(-tau, rho, z_exp),
            # y2 >= cp.quad_over_lin(y1, rho) + z_exp
            *[y2[j] >= cp.quad_over_lin(y1[j], rho[j]) + z_exp[j] for j in range(J)]
        ])
        # this is confirmed to agree with Cramer4-1 when we omit the m2 constraint
        # but we would like a way to show it agrees with Cramer4-2 when we omit the m1 constraint...
        perspective_terms = cp.sum([
            1/2 * (y2[j] / sigma**2 + tau[j])
            for j in range(J)
        ])
        rate_cost = cp.sum(perspective_terms)
        # 
        m1 = 0.0
        m2 = 1.0
        # moment constraints
        if use_12_m1_constraint:
            constraints.append(
                cp.sum(y1) + cp.sum(cp.multiply(rho, mu)) == m1,
            )
        constraints.append(
            # cp.sum([rho[j]*mu[j]**2 + y2[j] for j in range(J)]) == m2,
            cp.sum([rho[j]*mu[j]**2 + 2*mu[j]*y1[j] + y2[j] for j in range(J)]) == m2,
        )
    elif moment == 'gaussian':
        # (personal note: it is highly significant that w/o the info bound this gives the same result as Cramer Twelve. Prolly deviating to a gaussian is the cheapest anyways)
        #
        # D_KL(P||Q) = log(σ_Q/σ_P) + (σ_P^2 + (μ_P - μ_Q)^2) / (2σ_Q^2) - 1/2
        m_red = cp.Variable(J, name="m_red")  # m_j * rho_j
        s_red = cp.Variable(J, name="s_red")  # s_j * rho_j
        # likelihood 
        # for σ_Q=1 and μ_Q=μ_j we have
        # D_KL(P||Q) = - log(σ_P) + (σ_P^2 + (μ_P - μ_j)^2) / (2) - 1/2
        # = - 0.5 * log(s_j) + (s_j + (m_j - μ_j)^2) / (2) - 1/2
        # times rho_j
        # = -0.5ρ_j log(s_red/ρ_j) + 0.5ρ_j (s_j + (m_j - μ_j)^2) - 0.5ρ_j
        # up to a constant:
        # = -0.5ρ_j log(s_red/ρ_j) + 0.5ρ_j ((s_red/ρ_j) + ((m_red/ρ_j) - μ_j)^2)
        # = 0.5ρ_j log(ρ_j/s_red) + 0.5 s_red + 0.5 (m_red - μ_j*ρ_j)^2/ρ_j
        terms = [
            # rho D_KL(P_j || mu[j] centered 1-sigma gaussian)
            0.5*cp.rel_entr(rho[j], s_red[j]) + 0.5*s_red[j] + 0.5*cp.quad_over_lin(m_red[j] - mu[j]*rho[j], rho[j])
            for j in range(J)
        ]
        rate_cost = cp.sum(terms)
        constraints.append(
            cp.sum(m_red) == 0
        )
        # ρE[(m_j + ε_j)²] = ρ(m_j)^2 + ρ(s_j)
        # = (m_red)^2 / ρ + s_red
        constraints.append(
            cp.sum([cp.quad_over_lin(m_red[j], rho[j]) + s_red[j] for j in range(J)]) <= sigma**2
        )
        if gaussian_info_bound:
            constraints.extend([
                # rho D_KL(P_j || stdnormal) <= rho (-log(rho))
                # rho/2 is justified by the constant term we had previously omitted.
                0.5*cp.rel_entr(rho[j], s_red[j]) + 0.5*s_red[j] + 0.5*cp.quad_over_lin(m_red[j], rho[j]) - cp.entr(rho[j]) <= rho[j]/2
                for j in range(J)
            ])
        
    elif moment == 'indicator':
        pass
        # D = 100
        # bounds = [-np.inf,-4,-3,-2,-1,0,1,2,3,4,np.inf]
        bounds = [-np.inf,] + list(np.linspace(-10,10,20)) + [np.inf,]  # TODO: requires dx=1
        D = len(bounds) - 1
        Y = cp.Variable((J, D), name="Y")
        X_prior = np.empty((J, D))
        log_X_prior = np.empty((J, D))
        # bounds = np.array([stats.norm.isf(1 - d/D) for d in range(D+1)])
        print(f"Bounds: {bounds}")
        for j in range(J):
            cdf = lambda x: stats.norm.cdf(x, mu[j], sigma)
            logcdf = lambda x: stats.norm.logcdf(x, mu[j], sigma)
            for d in range(D):
                lower_end = bounds[d]
                upper_end = bounds[d+1]
                X_prior[j, d] = cdf(upper_end) - cdf(lower_end)
                # Let A = logcdf(upper), B = logcdf(lower)
                # We want log( exp(A) - exp(B) )
                # Since A > B, the term (B - A) is negative.
                # We use -expm1(B - A) which computes -(exp(B-A) - 1) = 1 - exp(B-A)
                diff = logcdf(lower_end) - logcdf(upper_end)
                log_X_prior[j, d] = logcdf(upper_end) + np.log(-np.expm1(diff))
        print(f"X_prior: {X_prior}")
        print(f"log_X_prior: {log_X_prior}")
        # for given j
        # ρ Σ_d x_d ln x_d/p_d
        # = Σ_d (y_d ln (y_d / ρ) - y_d ln p_d )
        # = Σ_d    rel_entr           linear
        rate_cost = 0
        for j in range(J):
            for d in range(D):
                # rate_cost += cp.rel_entr(Y[j,d], rho[j]) - Y[j,d]*log_X_prior[j,d]
                rate_cost += cp.rel_entr(Y[j,d], rho[j] * X_prior[j,d])
        # constraints
        # weighted average of x_j equals p_0
        for d in range(D):
            constraints.append(
                cp.sum([Y[j,d] for j in range(J)]) == X_prior[0,d]
            )
        # x_j is on the simplex
        for j in range(J):
            constraints.append(
                cp.sum([Y[j,d] for d in range(D)]) == rho[j]
            )


    else:
        raise ValueError(f"Moment {moment} not supported. Must be 1 or 2.")
    kl_cost = cp.sum(cp.rel_entr(rho, pi))
    objective = cp.Minimize(rate_cost + kl_cost)

    problem = cp.Problem(objective, constraints)

    # --- 5. Solve the Problem ---
    problem = cp.Problem(objective, constraints)
    print(f"Problem is DCP: {problem.is_dcp()}")
    # The problem is convex (DCP) and can be solved
    problem.solve()
    print(f"Problem status: {problem.status}")
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Problem status is {problem.status}. For sep={sep}, moment={moment}, use_12_m1_constraint={use_12_m1_constraint}")

    print(f"\nOptimal value: {problem.value:.4f}")

    # --- 6. Recover Optimal x_j and I_j ---
    # Recover x_j from the optimal rho_j and y_j
    # x_j = y_j / rho_j
    # x_optimal = y.value / rho.value

    # Calculate the optimal rate functions I_j
    # I_optimal = (x_optimal - mu)**2 / (2 * sigma**2)
    
    print("\n--- Optimal Solution ---")
    print(f"Separation: {sep}")
    print(f"Optimal rho_j (weights):\n{rho.value.round(4)}")
    # print(f"Optimal x_j (variables):\n{x_optimal.round(4)}")
    # print(f"Optimal I_j (rate functions):\n{I_optimal.round(4)}")
    if moment == 12:
        # x1
        x1 = y1.value / rho.value
        x2 = y2.value / rho.value
        print(f"Optimal x1_j:\n{x1.round(4)}")
        print(f"Optimal x2_j:\n{x2.round(4)}")

    if moment == 'gaussian':
        m, s = m_red.value / rho.value, s_red.value / rho.value
        print(f"Optimal m_j:\n{m.round(4)}")
        print(f"Optimal s_j:\n{s.round(4)}")
    
    # Verification of constraints and objective
    print("\n--- Verification ---")
    print(f"Sum of rho_j: {np.sum(rho.value).round(4)}")
    # print(f"Sum of y_j (rho_j * x_j): {np.sum(y.value).round(4)}")
    # print(f"Weighted objective sum: {np.sum(rho.value * I_optimal).round(4)}")

    if moment == 12:
        return rho.value, y1.value, y2.value
    elif moment == 'gaussian':
        return rho.value, m_red.value, s_red.value
    elif moment == 'indicator':
        return rho.value, Y.value, bounds
    return rho.value, None, None

def H_normal(sigma):
    return np.log(sigma) + 0.5 * np.log(2 * np.pi * np.e)
def H_biexponential(sigma):
    return np.log(sigma) + 0.5 * np.log(2) + 1

def shannon_entropy(pi):
    positive_pi = np.where(pi > 0, pi, 1e-100)
    return - np.sum(positive_pi * np.log(positive_pi))

def cramer4_H_X(sep, moment=2, use_12_m1_constraint=True, fix_mutual=True, gaussian_info_bound=False, plot_indicators=False):
    J = 2
    prior = np.array([0.5, 0.5])
    prior_ent = shannon_entropy(prior)
    rho, y1, y2 = find_post_rho(sep, moment, use_12_m1_constraint=use_12_m1_constraint, gaussian_info_bound=gaussian_info_bound)
    cond_ent = shannon_entropy(rho)
    if moment == 12 and fix_mutual:
        # I(X+; A+) = H(X+) - H(X+|A+)
        x1, x2 = y1 / rho, y2 / rho
        mutual = H_normal(1) - sum((H_normal(  (x2[j] - x1[j]**2)**0.5  )) * rho[j] for j in range(J))
        cond_ent -= mutual
    elif moment == 'gaussian' and fix_mutual:
        rho, m_red, s_red = rho, y1, y2
        m, s = m_red / rho, s_red / rho
        mutual = H_normal(1) - sum(H_normal(s[j]**0.5) * rho[j] for j in range(J))
        cond_ent -= mutual
    elif moment == 'indicator' and fix_mutual:
        rho, Y, bounds = rho, y1, y2
        bounds = np.array(bounds)
        mutual = H_normal(1) - sum(shannon_entropy(Y[j] / sum(Y[j])) * rho[j] for j in range(J))
        cond_ent -= mutual
        if plot_indicators:
            plt.figure(figsize=(11, 8))
            bound_midpoints = (bounds[1:-2] + bounds[2:-1])/2
            interval_densities1 = Y[0,1:-1] / sum(Y[0,1:-1])
            interval_densities2 = Y[1,1:-1] / sum(Y[1,1:-1])
            log_interval_densities1 = np.log(interval_densities1)
            log_interval_densities2 = np.log(interval_densities2)
            std_normal_logpdf = stats.norm.logpdf(bound_midpoints, 0, 1)
            plt.plot(bound_midpoints, log_interval_densities1, label='Indicator 0')
            plt.plot(bound_midpoints, log_interval_densities2, label='Indicator 1')
            plt.plot(bound_midpoints, std_normal_logpdf, label='Standard Normal')
            plt.plot(bound_midpoints, log_interval_densities1 - std_normal_logpdf, label='Indicator 0 - Standard Normal')
            plt.plot(bound_midpoints, log_interval_densities2 - std_normal_logpdf, label='Indicator 1 - Standard Normal')
            # plt.yscale('log')
            plt.xlim(-5., 5.)
            plt.ylim(-10, 2)
            plt.legend()
            plt.show()
    return prior_ent + H_normal(1) - cond_ent
#%%

cramer4_H_X(2., moment='indicator', plot_indicators=True)
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
    xs = np.linspace(-10*sigma, separation + 10*sigma, 400)
    vals = np.array([gmm_pdf(x, means, sigma, weights) for x in xs])
    return empirical_entropy(xs, vals)
def kolchinsky_ent(sep, dist_func, sigma=1.0, means=np.array([0, 1]), weights=np.array([0.5, 0.5])):
    # Kolchinsky & Tracey, Estimating Mixture Entropy with Pairwise Distances
    # H_D(X) := H(X|C) - Sum_i c_i ln( Sum_j c_j exp( - dist_func(p_i||p_j)
    if sep is not None:
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

plt.figure(figsize=(11, 8))
separations = np.linspace(0, 8, 100)
plt.plot(separations, [entropy_for_separation(x) for x in separations], label='Empirical')
# plt.plot(separations, [cramer4_H_X(x, moment=1) for x in separations], label='Cramer4-1')
# plt.plot(separations, [cramer4_H_X(x, moment=2) for x in separations], label='Cramer4-2')
# plt.plot(separations, [cramer4_H_X(x, moment=12, use_12_m1_constraint=False) for x in separations], label='Cramer4-12-2')
plt.plot(separations, [cramer4_H_X(x, moment=12) for x in separations], label='Cramer4-12')
# plt.plot(separations, [cramer4_H_X(x, moment=12, fix_mutual=False) for x in separations], label='Cramer4-12-no-mutual')
# plt.plot(separations, [cramer4_H_X(x, moment='indicator') for x in separations], label='Cramer4-Indicator')
# plt.plot(separations, [cramer4_H_X(x, moment='gaussian') for x in separations], label='Cramer4-Gaussian')
# plt.plot(separations, [cramer4_H_X(x, moment='gaussian', fix_mutual=False) for x in separations], label='Cramer4-Gaussian-no-mutual')
plt.plot(separations, [cramer4_H_X(x, moment='gaussian', gaussian_info_bound=True) for x in separations], label='Cramer4-Gaussian-info-bound')
# plt.plot(separations, [cramer4_H_X(x, moment='gaussian', fix_mutual=False, gaussian_info_bound=True) for x in separations], label='Cramer4-Gaussian-no-mutual-info-bound')
plt.plot(separations, [kolchinsky_ent(x, bhattacharyya_dist) for x in separations], label='Kolchinsky')
plt.hlines([H_normal(1), H_normal(1)+np.log(2)], xmin=0, xmax=8, color='red', linewidth=.5)
plt.legend()
plt.xlim(0, 8)
# plt.ylim(2.0, 2.15)
plt.show()

#%%
separations = np.linspace(0, 8, 100)
empirical = np.array([entropy_for_separation(x) for x in separations])
kolchinsky = np.array([kolchinsky_ent(x, bhattacharyya_dist) for x in separations])
cramer4_gauss_info = np.array([cramer4_H_X(x, moment='gaussian', gaussian_info_bound=True) for x in separations])
plt.plot(separations, np.abs(empirical - kolchinsky) / empirical, label='Kolchinsky rel. error')
plt.plot(separations, np.abs(empirical - cramer4_gauss_info) / empirical, label='Cramer4-Gaussian-info-bound rel. error')
plt.xlim(0, 8)
plt.ylim(0, 0.15)
plt.legend()
plt.show()
