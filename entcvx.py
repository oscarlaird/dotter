#%%

import time
import cvxpy as cp
import numpy as np
from scipy import stats
from scipy.special import ndtri

def compute_analytical_moments(xs, target_mu, target_sigma, r, tail):
    """
    Computes partial moments E[Y | condition] * P(condition) vectorized.
    Returns F_x_vals and slope_vals (quantiles).
    """
    slope_vals = np.zeros_like(xs)
    F_x_vals = np.zeros_like(xs)

    if r == 1:
        # --- Normal Distribution Case ---
        # Y = X (Normal)
        mu, sigma = target_mu, target_sigma
        
        if tail == 0:
            # Lower tail: P(Y < q) = x
            z_scores = ndtri(xs) 
            slope_vals = mu + sigma * z_scores
            
            # Formula: E[Y | Y < q] * P(Y < q) = mu * x - sigma * pdf(z)
            # Note: stats.norm.pdf(z) = exp(-z^2/2) / sqrt(2pi)
            pdf_vals = np.exp(-0.5 * z_scores**2) / np.sqrt(2 * np.pi)
            F_x_vals = mu * xs - sigma * pdf_vals
            
        elif tail == 1:
            # Upper tail: P(Y > q) = x  =>  P(Y < q) = 1 - x
            z_scores = ndtri(1.0 - xs)
            slope_vals = mu + sigma * z_scores
            
            # Formula: E[Y | Y > q] * P(Y > q) = mu * x + sigma * pdf(z)
            pdf_vals = np.exp(-0.5 * z_scores**2) / np.sqrt(2 * np.pi)
            F_x_vals = mu * xs + sigma * pdf_vals

    elif r == 2:
        # --- Non-Central Chi-Squared Case ---
        # Y = X^2 (Non-central Chi-sq)
        # y_dist is stats.ncx2
        df = 1.0
        # safety check for zero division, though target_sigma shouldn't be 0
        nc = (target_mu / target_sigma) ** 2
        scale = target_sigma ** 2
        
        if tail == 0:
            # Lower tail
            # q is the actual value in the scaled domain
            q_vals = stats.ncx2.ppf(xs, df=df, nc=nc, scale=scale)
            slope_vals = q_vals
            
            # Convert q back to "standard" scale for the formula
            q_std = q_vals / scale
            
            # Formula: Integral_0^q t * pdf(t) dt for ncx2(df, nc) 
            #        = df * CDF(q, df+2) + nc * CDF(q, df+4)
            # We perform this on the standard variable, then multiply by scale
            term1 = df * stats.ncx2.cdf(q_std, df=df+2, nc=nc)
            term2 = nc * stats.ncx2.cdf(q_std, df=df+4, nc=nc)
            
            F_x_vals = scale * (term1 + term2)
            
        elif tail == 1:
            # Upper tail
            q_vals = stats.ncx2.isf(xs, df=df, nc=nc, scale=scale)
            slope_vals = q_vals
            q_std = q_vals / scale
            
            # Total Mean of standard ncx2 is (df + nc)
            total_mean_std = df + nc
            
            # Upper integral = Total Mean - Lower Integral
            term1 = df * stats.ncx2.cdf(q_std, df=df+2, nc=nc)
            term2 = nc * stats.ncx2.cdf(q_std, df=df+4, nc=nc)
            lower_integral_std = term1 + term2
            
            F_x_vals = scale * (total_mean_std - lower_integral_std)
            
    return F_x_vals, slope_vals

def find_post_rho(mu, pi, sigmas, target_index=0, method='gaussian', method_kwargs=None, verbose=False):
    # Infer the posterior distributions rho and X|A_i,[Distributional Stats of X]
    J, D = mu.shape
    target_mu = mu[target_index]
    target_sigma = sigmas[target_index]
    method_kwargs = method_kwargs or {}
    assert len(pi) == J
    # def find_post_rho(sep, method=2, use_12_m1_constraint=True, gaussian_info_bound=False):
    # J = 2
    # TODO: for now we assume homoskedastic and isotropic, so sigma can just be a constant
    rho = cp.Variable(J, nonneg=True, name="rho")

    t_start = time.time()
    t_fx = 0.0

    rate_cost = None
    moment_constraint = None
    constraints = [
        cp.sum(rho) == 1,
    ]
    if method == 'gaussian':
        if D > 1:
            raise ValueError(f"Gaussian method only supports D=1. For D={D}, use method='diag_gaussian'.")
        # (personal note: it is highly significant that w/o the info bound this gives the same result as Cramer Twelve. Prolly deviating to a gaussian is the cheapest anyways)
        #
        # D_KL(P||Q) = log(σ_Q/σ_P) + (σ_P^2 + (μ_P - μ_Q)^2) / (2σ_Q^2) - 1/2
        m_red = cp.Variable(J, name="m_red")  # m_j * rho_j
        S_RED = cp.Variable(name="S_RED")
        # s_red = np.ones(J) * S_RED
        s_red = cp.Variable(J, name="s_red")  # s_j * rho_j
        # likelihood 
        # for μ_Q=μ_j we have
        # D_KL(P||Q) = - log(σ_P) + (σ_P^2 + (μ_P - μ_j)^2) / (2σ_Q^2) - 1/2 + log(σ_Q)
        # = - 0.5 * log(s_j) + (s_j + (m_j - μ_j)^2) / (2σ_Q^2) - 1/2 + log(σ_Q)
        # times rho_j
        # = -0.5ρ_j log(s_red/ρ_j) + (0.5/σ_Q^2)ρ_j (s_j + (m_j - μ_j)^2) + (-0.5 + log(σ_Q))ρ_j
        # up to a constant:
        # = -0.5ρ_j log(s_red/ρ_j) + (0.5/σ_Q^2)ρ_j ((s_red/ρ_j) + ((m_red/ρ_j) - μ_j)^2)
        # = 0.5ρ_j log(ρ_j/s_red) + (0.5/σ_Q^2)s_red + (0.5/σ_Q^2) (m_red - μ_j*ρ_j)^2/ρ_j
        terms = [
            # rho D_KL(P_j || origin of component)
            0.5*cp.rel_entr(rho[j], s_red[j]) + 0.5*s_red[j]/origin_sigma**2 + 0.5*cp.quad_over_lin(m_red[j] - origin_mu[0]*rho[j], rho[j])/origin_sigma**2
            for j, (origin_sigma, origin_mu) in enumerate(zip(sigmas, mu))
        ]
        rate_cost = cp.sum(terms)
        constraints.append(
            cp.sum(m_red) == target_mu[0]
        )
        # ρE[(m_j + ε_j)²] = ρ(m_j)^2 + ρ(s_j)
        # = (m_red)^2 / ρ + s_red
        constraints.append(
            # TODO: We should be able to rewrite this with an equality constraint
            cp.sum([cp.quad_over_lin(m_red[j], rho[j]) + s_red[j] for j in range(J)]) <= target_mu[0]**2 + target_sigma**2
        )
        if method_kwargs.get('gaussian_info_bound', True):
            constraints.extend([
                # rho D_KL(P_j || target component) <= rho (-log(rho))
                # rho/2 is justified by the constant term we had previously omitted.
                0.5*cp.rel_entr(rho[j], s_red[j])
                + 0.5*s_red[j]/target_sigma**2
                + 0.5*cp.quad_over_lin(m_red[j] - target_mu[0]*rho[j], rho[j])/target_sigma**2
                + rho[j]*(-0.5 + np.log(target_sigma))
                <= cp.entr(rho[j])
                for j in range(J)
            ])
        if method_kwargs.get('nest_parabolas', True):
            # We observe the distribution of x, z
            # Enforcing sum_i f_i <= z would be intractable as well as infeasible
            # However we can require each component to be less: f_i <= z
            # This is the requirement (in log space) that the weighted normal ρN lie beneath the std normal
            # i.e. that the parabolas are nested
            # The condition is that,
            # ln(ρ/σ) + μ²/(2 - 2σ²) <= 0
            # First we need to scale about the target component
            mc_red = [(m_red[j] - target_mu[0] * rho[j]) / target_sigma for j in range(J)]
            sc_red = s_red / target_sigma**2  # recall that s_red = σ_red^2 ρ
            # ln(ρ/σ) + μ²/(2 - 2σ²) <= 0
            # ln(ρ/σ) + μ²/(2 - 2σ²) <= 0
            # ln(ρ) - ln(σ) + μ²/(2 - 2σ²) <= 0
            # ln(ρ) - 0.5*ln(σ²) + μ²/(2 - 2σ²) <= 0
            # ln(ρ) - 0.5*ln(s/ρ) + (m/ρ)²/(2 - 2(s/ρ)) <= 0
            # ln(ρ) + 0.5*ln(ρ/s) + (m²/ρ)/(2ρ - 2s) <= 0
            # multiply by rho
            # ρ ln(ρ) + 0.5*ρ ln(ρ/s) + m²/(2ρ - 2s) <= 0
            constraints.extend([
                - cp.entr(rho[j])
                + 0.5*cp.rel_entr(rho[j], sc_red[j])
                + cp.quad_over_lin(mc_red[j], 2.0*rho[j] - 2.0*sc_red[j])
                # <= 0
                # <= np.log(2) * rho[j]
                <= 0.5 * rho[j]
                for j in range(J)
            ])
            constraints.extend([
                sc_red[j] <= rho[j]  # σ^2 <= 1 to fall beneath std normal
                for j in range(J)
            ])
    elif method == 'diag_gaussian':
        m_red = cp.Variable((J,D), name="m_red")
        s_red = cp.Variable((J,D), name="s_red")
        # The D_KL between diagonal gaussians is the sum of the divergences for each dimension
        # Prior penalty: divergence from component's origin
        terms = [
            # rho D_KL(P_j || origin of component)
            0.5*cp.rel_entr(rho[j], s_red[j,d]) + 0.5*s_red[j,d]/origin_sigma**2 + 0.5*cp.quad_over_lin(m_red[j,d] - origin_mu[d]*rho[j], rho[j])/origin_sigma**2
            for j,(origin_sigma, origin_mu) in enumerate(zip(sigmas, mu))
            for d in range(D)
        ]
        rate_cost = cp.sum(terms)
        # moment constraints in each dimension
        # 1st moment
        constraints.extend([
            cp.sum(m_red[:, d]) == target_mu[d]
            for d in range(D)
        ])
        # 2nd moment
        constraints.extend([
            # TODO: We should be able to rewrite this with an equality constraint
            cp.sum([cp.quad_over_lin(m_red[j,d], rho[j]) + s_red[j,d] for j in range(J)]) <= target_mu[d]**2 + target_sigma**2
            for d in range(D)
        ])
        if method_kwargs.get('gaussian_info_bound', True):
            constraints.extend([
                # rho D_KL(P_j || stdnormal) <= rho (-log(rho))
                # rho/2 is justified by the constant term we had previously omitted.
                cp.sum([
                    0.5*cp.rel_entr(rho[j], s_red[j,d])
                    + 0.5*s_red[j,d]/target_sigma**2
                    + 0.5*cp.quad_over_lin(m_red[j,d] - target_mu[d]*rho[j], rho[j])/target_sigma**2
                    + rho[j]*(-0.5 + np.log(target_sigma))
                    for d in range(D)
                ]) <= cp.entr(rho[j])
                for j in range(J)
            ])
        # in general we want to return rho and our characterization of the posterior component distributions
    elif method == 1:
        m = 0
        perspective_terms = cp.sum([
            cp.quad_over_lin(y[j], rho[j]) 
            for j in range(J)
        ])
        rate_cost = (1 / (2 * target_sigma**2)) * cp.sum(perspective_terms)
        #
        moment_constraint = cp.sum(y) + cp.sum(cp.multiply(rho, mu)) == m
        constraints.append(moment_constraint)
    elif method == 2:
        assert D == 1, f"D={D} != 1"
        y = cp.Variable(J, name="y")
        m = target_sigma**2
        # Rate for X^2
        # I(x) = 1/2 [x/sigma^2 - 1 - log(x/sigma^2)]
        # ρI(x) = ρI(y/ρ)
        #       = 1/2 [ y/σ^2 - ρ + ρ log(ρ/y) + log(sigma^2)]
        perspective_terms = cp.sum([
            1/2 * (y[j]/target_sigma**2 - rho[j])
            # + 1/2 * cp.log(sigma**2)  # constant, so ignored
            + 1/2 * cp.rel_entr(rho[j], y[j])
            for j in range(J)
        ])
        rate_cost = cp.sum(perspective_terms)
        # 
        # TODO!: Is this wrong?! We don't know that E[X]=0 when we put this constraint on E[X^2]
        # But I tend to think that this is the most likely (in fact, easy to prove from cvxity)
        moment_constraint = cp.sum([rho[j]*mu[j,0]**2 + y[j] for j in range(J)]) == m
        constraints.append(moment_constraint)
    elif method == 12:
        m1 = target_mu[0]
        m2 = target_sigma**2
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
            1/2 * (y2[j] / target_sigma**2 + tau[j])
            for j in range(J)
        ])
        rate_cost = cp.sum(perspective_terms)
        # moment constraints
        # if use_12_m1_constraint:
        #     constraints.append(
        #         cp.sum(y1) + cp.sum(cp.multiply(rho, mu)) == m1,
        #     )
        constraints.append(
            # cp.sum([rho[j]*mu[j]**2 + y2[j] for j in range(J)]) == m2,
            cp.sum([rho[j]*mu[j]**2 + 2*mu[j]*y1[j] + y2[j] for j in range(J)]) == m2,
        )
    elif method == 'indicator':
        pass
        # D = 100
        # bounds = [-np.inf,-4,-3,-2,-1,0,1,2,3,4,np.inf]
        bounds = [-np.inf,] + list(np.linspace(-10,10,200)) + [np.inf,]  # TODO: requires dx=1
        D = len(bounds) - 1
        Y = cp.Variable((J, D), name="Y")
        X_prior = np.empty((J, D))
        log_X_prior = np.empty((J, D))
        # bounds = np.array([stats.norm.isf(1 - d/D) for d in range(D+1)])
        for j in range(J):
            cdf = lambda x: stats.norm.cdf(x, mu[j], target_sigma)
            logcdf = lambda x: stats.norm.logcdf(x, mu[j], target_sigma)
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
    elif method == 'chebyshev_gaussian':
        assert D == 1, f"D={D} != 1. Chebyshev Gaussian only supports 1-dimensional distributions."
        target_mu = target_mu[0]
        
        # t_start and t_fx initialized at function start

        R = 3  # number of statistic observations
        # we pad a dummy α0 just so that the indexing matches the standard moment naming
        α_ = cp.Variable((R, J), name="AlphaReduced")   # α_ = α * rho
        # 1. Objective (aka prior)
        rate_cost = 0
        for j in range(J):
            component_origin_mu = mu[j][0]
            component_origin_sigma = sigmas[j]
            # + Cross-entropy
            # Cross-entropy(P||Q) = (σp² + (μp - μq)²) / (2σq²) + 0.5 * log(2π σq²)
            # and σp² = α2 - α1^2; μp = α1
            # = (α2 - α1² + α1² - 2*α1*μq + μq²) / (2σq²) + 0.5 * log(2π σq²)
            # = (α2 - 2*α1*μq + μq²) / (2σq²) + 0.5 * log(2π σq²)
            cross_ent = 0
            cross_ent += α_[2,j] - 2*α_[1,j]*component_origin_mu + rho[j]*component_origin_mu**2  # linear
            cross_ent /= 2*component_origin_sigma**2
            cross_ent += rho[j] * 0.5 * np.log(2 * np.pi * component_origin_sigma**2)
            # - Entropy
            # H(P) = ½ log(2πe σp²)
            # = ½ log(2πe (α2 - α1²))
            # after applying the perspective transformation:
            # = ½ ρ log(2πe (α2/ρ - (α1/ρ)²))
            # = ½ ρ log(2πe) + ½ ρ log((α2 - α1²/ρ) / ρ)
            # = ½ ρ log(2πe) - ½ ρ log(ρ / (α2 - α1²/ρ))
            ent = 0.5 * np.log(2 * np.pi * np.e) * rho[j]
            ent -= 0.5 * cp.rel_entr(rho[j], α_[2,j] - cp.quad_over_lin(α_[1,j], rho[j]))
            # D_KL
            rate_cost += cross_ent - ent
            

        # 2. Constraints (aka likelihood)
        # Constraint: α_[2,j] must be non-negative (since it represents E[X^2]*rho)
        constraints.append(α_[2,:] >= 0)
        assert J <= 4, f"J={J} > 4. Chebyshev Gaussian creates 2^J constraints. Hence it is not feasible for J > 4."
        import itertools
        # Create kappas matrix (excluding all-zeros)
        kappas_list = [k for k in itertools.product([0, 1], repeat=J) if sum(k) > 0]
        K_count = len(kappas_list)
        kappas_matrix = np.array(kappas_list)  # Shape (K, J)

        # Pre-calculate vectorized rho sums: total_rho_vec[k] = sum(rho[j] for j where kappa[k][j]==1)
        # Shape (K, 1) after reshape
        total_rho_vec = (kappas_matrix @ rho).reshape((K_count, 1))

        # Constants for piecewise linear approx
        logits = method_kwargs.get('logitspace', np.linspace(-5, 5, 10))
        xs = 1 / (1 + np.exp(-logits))  # Shape (N,)
        N_points = len(xs)
        xs_row = xs.reshape((1, N_points))

        for r in range(R):
            if r == 0:
                continue
            
            # Vectorized LHS: sum(α_[r,j] for j where kappa[k][j]==1)
            # Shape (K, 1)
            lhs_vec = (kappas_matrix @ α_[r, :]).reshape((K_count, 1))

            # TAIL
            for tail in range(2):
                # Precompute distribution constants for all x
                # We do this once per (r, tail) instead of per kappa
                t_fx_start = time.time()
                
                # Optimized vectorized call
                F_x_vals, slope_vals = compute_analytical_moments(xs, target_mu, target_sigma, r, tail)
                
                t_fx += time.time() - t_fx_start
                
                # Reshape for broadcasting
                # F_x: (1, N)
                # slope: (1, N)
                F_x_row = F_x_vals.reshape((1, N_points))
                slope_row = slope_vals.reshape((1, N_points))
                
                # Construct vectorized RHS
                # line = F_x + slope * (total_rho - x)
                #      = (F_x - slope * x) + slope * total_rho
                # RHS matrix shape: (K, N)
                # total_rho_vec is (K, 1), slope_row is (1, N) -> total_rho_vec @ slope_row is (K, N)
                
                constant_part = F_x_row - slope_row * xs_row
                linear_part = total_rho_vec @ slope_row
                
                rhs_matrix = constant_part + linear_part
                
                # Broadcast LHS to (K, N) implicitly during comparison or explicit reshape?
                # lhs_vec is (K, 1). Comparison with (K, N) broadcasts correctly in CVXPY.
                
                if tail == 0:
                    constraints.append(lhs_vec >= rhs_matrix - 1e-6)
                elif tail == 1:
                    constraints.append(lhs_vec <= rhs_matrix + 1e-6)

    else:
        raise ValueError(f"method {method} not supported. Must be 'gaussian', 'diag_gaussian', 'indicator', 'chebyshev_gaussian', or '12'.")

    kl_cost = cp.sum(cp.rel_entr(rho, pi))
    objective = cp.Minimize(rate_cost + kl_cost)
    problem = cp.Problem(objective, constraints)
    
    t_compile_end = time.time()
    # try:
    #     problem.solve()
    # except cp.error.SolverError as e:
    #     print(f"Solver failed with error: {e}")
    #     print("Retrying with SCS...")
    problem.solve(solver=cp.SCS)
    t_solve_end = time.time()
    
    if method == 'chebyshev_gaussian' and verbose:
        t_setup = t_compile_end - t_start - t_fx
        t_solve = t_solve_end - t_compile_end
        print(f"[Chebyshev] F_x: {t_fx:.4f}s, Compile: {t_setup:.4f}s, Solve: {t_solve:.4f}s")

    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Problem status is {problem.status}. For mu={mu}, pi={pi}, method={method}")

    if method == 12:
        return rho.value, {'y1': y1.value, 'y2': y2.value}
    elif method == 'gaussian':
        return rho.value, {'m_red': m_red.value, 's_red': s_red.value}
    elif method == 'diag_gaussian':
        return rho.value, {'m_red': m_red.value, 's_red': s_red.value}
    elif method == 'chebyshev_gaussian':
        return rho.value, {'AlphaReduced': α_.value}
    elif method == 'indicator':
        return rho.value, {'Y': Y.value, 'bounds': bounds}
    return rho.value, {}

def fix_mutual(rho, stats, method, plot_indicators=False):
    # Evaluate H(X_i | A^(D+), [Distributional Stats of X]) and A is distributed according to rho
    J = len(rho)
    if method == 12:
        y1, y2 = stats['y1'], stats['y2']
        x1, x2 = y1 / rho, y2 / rho
        return sum((H_normal(  (x2[j] - x1[j]**2)**0.5  )) * rho[j] for j in range(J))
    elif method == 'gaussian':
        # return 0
        m_red, s_red = stats['m_red'], stats['s_red']
        m, s = m_red / rho, s_red / rho
        return sum(H_normal(s[j]**0.5) * rho[j] for j in range(J))
    elif method == 'diag_gaussian':
        J, D = stats['m_red'].shape
        # assert (J,) == rho.shape, f"rho.shape={rho.shape} != (J,)={(J,)}"
        # assert (J, D) == stats['s_red'].shape, f"stats['s_red'].shape={stats['s_red'].shape} != (J, D)={(J, D)}"
        m_red, s_red = stats['m_red'], stats['s_red']
        m, s = m_red / rho[:, None], s_red / rho[:, None]
        s = s**.5
        assert (J, D) == s.shape, f"s.shape={s.shape} != (J, D)={(J, D)}"
        return sum(H_normal(s[j,d]) * rho[j] for j in range(J) for d in range(D))
    elif method == 'chebyshev_gaussian':
        # return 0
        α_ = stats['AlphaReduced']
        a1_ = α_[1, :]
        a2_ = α_[2, :]
        a1, a2 = a1_ / rho, a2_ / rho
        variances = a2 - a1**2
        variances = np.maximum(variances, 1e-10)
        assert (J,) == a1.shape, f"a1.shape={a1.shape} != (J)={(J,)}"
        assert (J,) == a2.shape, f"a2.shape={a2.shape} != (J)={(J,)}"
        assert (J,) == variances.shape, f"variances.shape={variances.shape} != (J)={(J,)}"
        return sum(H_normal(variances[j]**0.5) * rho[j] for j in range(J))
    elif method == 'indicator':
        Y, bounds = stats['Y'], stats['bounds']
        bounds = np.array(bounds)
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
            plt.plot(bound_midpoints, np.log(10*rho[0]*np.exp(log_interval_densities1) + 10*rho[1]*np.exp(log_interval_densities2)), label='Indicator 0 + Indicator 1')
            # plt.yscale('log')
            plt.xlim(-5., 5.)
            plt.ylim(-10, 2)
            plt.legend()
            plt.show()
        return sum(shannon_entropy(Y[j] / sum(Y[j])) * rho[j] for j in range(J))

def H_normal(sigma):
    return np.log(sigma) + 0.5 * np.log(2 * np.pi * np.e)
def H_biexponential(sigma):
    return np.log(sigma) + 0.5 * np.log(2) + 1

def shannon_entropy(pi):
    positive_pi = np.where(pi > 0, pi, 1e-100)
    return - np.sum(positive_pi * np.log(positive_pi))

def cramer4(mu, pi, sigmas, method='gaussian', method_kwargs={}, correction_order=None, verbose=False):
    J, D = mu.shape
    H_X_ests = []
    for i in range(J):
        # H(X) = E_i[  H(A) + H(ε_i) - H(A|D_i) + H(X|i) - H(X_n | A, D_i)    ] where D_i is the observation of the distributional statistics
        if verbose:
            if method_kwargs is None:
                method_kwargs = {}
            method_kwargs['verbose'] = True
        rho, stats = find_post_rho(mu, pi, sigmas, target_index=i, method=method, method_kwargs=method_kwargs, verbose=verbose)
        H_X_given_D = H_normal(sigmas[i])*D
        if verbose:
            print(f"Rho: {rho}")
            if method == 'gaussian':
                m_red, s_red = stats['m_red'], stats['s_red']
                m, s = m_red / rho, s_red / rho
                s = s**.5
                print(f"m: {m}")
                print(f"s: {s}")
        if correction_order:  # TODO: this correction only works if s_j is homoskedastic
            if method == 'gaussian':
                m_red, s_red = stats['m_red'], stats['s_red']
                m, s = m_red / rho, s_red / rho
                s = s**.5
                m = np.array([[m[k]] for k in range(J)])
                H_X_given_D = cramer4(m, rho, s, method='gaussian', correction_order=correction_order-1)
            elif method == 'chebyshev_gaussian':
                α_ = stats['AlphaReduced']
                a1_ = α_[1, :]
                a2_ = α_[2, :]
                a1, a2 = a1_ / rho, a2_ / rho
                variances = a2 - a1**2
                variances = np.maximum(variances, 1e-10)
                variances = variances**.5
                m = np.array([[a1[k]] for k in range(J)])
                H_X_given_D = cramer4(m, rho, variances, method='chebyshev_gaussian', correction_order=correction_order-1)
            else:
                raise NotImplementedError
        est = 0
        est += shannon_entropy(pi)
        est += H_normal(sigmas[i])*D
        est -= shannon_entropy(rho)
        est += H_X_given_D
        est -= fix_mutual(rho, stats, method)
        H_X_ests.append(est)
    weighted_H_X_ests = np.array(H_X_ests) * pi
    return np.sum(weighted_H_X_ests)

def cube_cramer4(sep, method='gaussian', method_kwargs=None, correction_order=None, verbose=False):
    print(f"Calculating cube Cramer4 for separation {sep} with method {method}")
    mu = np.array([[0.],
                   [sep]])
    pi = np.array([0.5, 0.5])
    sigmas = np.ones(2) * 1.0
    return cramer4(mu, pi, sigmas, method=method, method_kwargs=method_kwargs, correction_order=correction_order, verbose=verbose)

# def square_cramer4(sep, method='diag_gaussian', method_kwargs=None, correction_order=None):
#     # theoretically this allows each posterior distribution to be a two-component gaussian mixture,
#     # but in practice it has no effect
#     mu = np.array([[0., 0.],
#                    [sep, 0.],
#                    [0., sep],
#                    [sep, sep]])
#     pi = np.array([0.25, 0.25, 0.25, 0.25])
#     sigmas = np.ones(4) * 1.0
#     ent = cramer4(mu, pi, sigmas, method=method, method_kwargs=method_kwargs, correction_order=correction_order)
#     return ent/2.0

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
def entropy_for_separation(separation, sigma=1.0):
    means = np.array([0, separation])
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

if __name__ == "__main__":
    plt.figure(figsize=(11, 8))
    separations = np.linspace(0, 7, 70)
    # plt.plot(separations, [cube_cramer4(x, method=12) for x in separations], label='Cramer4-12')
    # plt.plot(separations, [cramer4_H_X(x, method='indicator') for x in separations], label='Cramer4-Indicator')
    # plt.plot(separations, [square_cramer4(x, method='diag_gaussian') for x in separations], label='Square-Cramer4-Diag-Gaussian')
    # plt.plot(separations, [cube_cramer4(x, method='gaussian', verbose=True, method_kwargs={'gaussian_info_bound': True, 'nest_parabolas': True}) for x in separations], label='Cube-Cramer4-Gaussian')
    # plt.plot(separations, [cube_cramer4(x, method='gaussian', verbose=True, method_kwargs={'gaussian_info_bound': False, 'nest_parabolas': True}, correction_order=0) for x in separations], label='Cube-Cramer4-Gaussian-no-info-bound')
    # plt.plot(separations, [cube_cramer4(x, method='gaussian', verbose=True, method_kwargs={'gaussian_info_bound': False, 'nest_parabolas': True}, correction_order=1) for x in separations], label='Cube-Cramer4-Gaussian-no-info-bound')
    # plt.plot(separations, [cube_cramer4(x, method='gaussian', verbose=True, method_kwargs={'gaussian_info_bound': True, 'nest_parabolas': False}) for x in separations], label='Cube-Cramer4-Gaussian-no-nest-parabolas')
    plt.plot(separations, [cube_cramer4(x, method='gaussian', verbose=True, method_kwargs={'gaussian_info_bound': False, 'nest_parabolas': False}) for x in separations], label='Cube-Cramer4-Gaussian-no-info-bound-no-nest-parabolas')
    plt.plot(separations, [entropy_for_separation(x, 1) for x in separations], label='Empirical')
    plt.plot(separations, [cube_cramer4(x, method='chebyshev_gaussian') for x in separations], label='CHEBYSHEV')
    plt.plot(separations, [cube_cramer4(x, method='chebyshev_gaussian', method_kwargs={'logitspace': np.linspace(-10,10,400)}) for x in separations], label='CHEBYSHEV-logitspace')
    plt.plot(separations, [cube_cramer4(x, method='chebyshev_gaussian', method_kwargs={'logitspace': np.linspace(-5,5,100)}) for x in separations], label='CHEBYSHEV-logitspace')
    plt.plot(separations, [cube_cramer4(x, method='chebyshev_gaussian', method_kwargs={'logitspace': np.linspace(-10,10,100)}) for x in separations], label='CHEBYSHEV-logitspace')
    # plt.plot(separations, [cube_cramer4(x, method='chebyshev_gaussian', verbose=True, correction_order=1) for x in separations], label='CHEBYSHEV-correction-1')
    # plt.plot(separations, [cube_cramer4(x, method='gaussian', correction_order=1) for x in separations], label='Cramer4-Gaussian-correction')
    # plt.plot(separations, [cramer4_H_X(x, method='gaussian', fix_mutual=False) for x in separations], label='Cramer4-Gaussian-no-mutual')
    # plt.plot(separations, [cramer4_H_X(x, method='gaussian', gaussian_info_bound=True) for x in separations], label='Cramer4-Gaussian-info-bound')
    # plt.plot(separations, [cramer4_H_X(x, method='gaussian', fix_mutual=False, gaussian_info_bound=True) for x in separations], label='Cramer4-Gaussian-no-mutual-info-bound')
    plt.plot(separations, [kolchinsky_ent(x, bhattacharyya_dist, 1) for x in separations], label='Kolchinsky')
    plt.hlines([H_normal(1), H_normal(1)+np.log(2)], xmin=0, xmax=8, color='red', linewidth=.5)
    plt.legend()
    # plt.ylim(2.0, 2.15)
    plt.show()

    #%%
    separations = np.linspace(0, 7.5, 100)
    empirical = np.array([entropy_for_separation(x) for x in separations])
    kolchinsky = np.array([kolchinsky_ent(x, bhattacharyya_dist) for x in separations])
    cramer4_chebyshev = np.array([cube_cramer4(x, method='chebyshev_gaussian', method_kwargs={'logitspace': np.linspace(-10,10,100)}) for x in separations])
    plt.plot(separations, np.abs(empirical - kolchinsky) / empirical, label='Kolchinsky rel. error')
    plt.plot(separations, np.abs(empirical - cramer4_chebyshev) / empirical, label='Cramer4-Chebyshev rel. error')
    plt.xlim(0, 8)
    plt.ylim(0, 0.15)
    plt.legend()
    plt.show()
