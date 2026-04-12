import torch
import math

# (x, P) pairs
dummy_data = [
    (0.900, 1.1),
    (0.950, 1.1),
    (0.900, 1.1),
    (0.950, 1.1),
    (1.100, 1.3),
    (1.150, 1.3),
]

# reasonable priors:
# m is reasonably -150ms to 150ms
mu_m_prior = 0.0
# The previous value was 80, but based on the context of 'ms' and the data being in seconds, 
# 80ms is 0.08 seconds. We fix it here.
sigma_m_prior = 0.080 
# std dev is reasonably 20ms to 200ms
# var is .02^2 to .2^2
# log var is -7.8 to -3.2
# s denotes log variance
mu_s_prior = -5.5
sigma_s_prior = 1.0
# rho is reasonably 0.01 to 0.20
# which 95% interval is given by:
# alpha = 2.5, beta = 25.0
# a = log(alpha), b = log(beta) 
a_prior = math.log(2.5)
b_prior = math.log(25.0)

# ALGORITHM:

def J(params, x, P, prior_params):
    mu_m_q = params[0]
    sigma_m_q = torch.exp(params[1])
    mu_s_q = params[2]
    sigma_s_q = torch.exp(params[3])
    a_q = params[4]
    b_q = params[5]
    
    alpha_q = torch.exp(a_q)
    beta_q = torch.exp(b_q)
    
    mu_m_p = prior_params[0]
    sigma_m_p = prior_params[1]
    mu_s_p = prior_params[2]
    sigma_s_p = prior_params[3]
    alpha_p = torch.exp(prior_params[4])
    beta_p = torch.exp(prior_params[5])

    # Expected log rho and log(1-rho)
    E_log_rho = torch.digamma(alpha_q) - torch.digamma(alpha_q + beta_q)
    E_log_1_minus_rho = torch.digamma(beta_q) - torch.digamma(alpha_q + beta_q)
    
    # L_k for k in {-1, 0, 1}
    L_k_terms = []
    for k in [-1, 0, 1]:
        term = E_log_1_minus_rho - 0.5 * math.log(2 * math.pi) - 0.5 * mu_s_q \
               - 0.5 * torch.exp(-mu_s_q + 0.5 * sigma_s_q**2) * ((x - mu_m_q - k * P)**2 + sigma_m_q**2)
        L_k_terms.append(term)
    L_k_stack = torch.stack(L_k_terms)
    
    term1 = E_log_rho - math.log(P)
    
    # logsumexp over the 4 modes (outlier + 3 normal modes)
    likelihood_bound = torch.logsumexp(torch.cat([term1.unsqueeze(0), L_k_stack]), dim=0)
    
    # KL divergence terms
    kl_m = torch.log(sigma_m_p / sigma_m_q) + (sigma_m_q**2 + (mu_m_q - mu_m_p)**2) / (2 * sigma_m_p**2) - 0.5
    kl_s = torch.log(sigma_s_p / sigma_s_q) + (sigma_s_q**2 + (mu_s_q - mu_s_p)**2) / (2 * sigma_s_p**2) - 0.5
    
    log_B_p = torch.lgamma(alpha_p) + torch.lgamma(beta_p) - torch.lgamma(alpha_p + beta_p)
    log_B_q = torch.lgamma(alpha_q) + torch.lgamma(beta_q) - torch.lgamma(alpha_q + beta_q)
    
    kl_beta = log_B_p - log_B_q \
              + (alpha_q - alpha_p) * torch.digamma(alpha_q) \
              + (beta_q - beta_p) * torch.digamma(beta_q) \
              - (alpha_q + beta_q - alpha_p - beta_p) * torch.digamma(alpha_q + beta_q)
              
    elbo = likelihood_bound - kl_m - kl_s - kl_beta
    return elbo

def optimize_online(x, P, prior_params):
    q_params = torch.tensor([
        prior_params[0],
        math.log(prior_params[1]),
        prior_params[2],
        math.log(prior_params[3]),
        prior_params[4],
        prior_params[5]
    ], dtype=torch.float64)
    
    def loss_fn(p):
        return -J(p, x, P, prior_params)
    
    for i in range(20):
        loss = loss_fn(q_params)
        grad = torch.autograd.functional.jacobian(loss_fn, q_params)
        hessian = torch.autograd.functional.hessian(loss_fn, q_params)
        
        # Ensure Hessian is positive definite for Newton's method to guarantee a descent direction
        evals, evecs = torch.linalg.eigh(hessian)
        evals = torch.clamp(evals, min=1e-4)
        hessian_pd = evecs @ torch.diag(evals) @ evecs.T
        
        step = torch.linalg.solve(hessian_pd, -grad)
            
        # Backtracking line search to ensure loss decreases
        alpha = 1.0
        c = 1e-4
        while alpha > 1e-6:
            new_params = q_params + alpha * step
            new_loss = loss_fn(new_params)
            if new_loss <= loss + c * alpha * torch.dot(grad, step):
                break
            alpha *= 0.5
            
        q_params = q_params + alpha * step
        
        if torch.norm(alpha * step) < 1e-6:
            print(f"Converged in {i+1} iterations. ELBO: {-loss.item():.4f}")
            break
    else:
        print(f"Did not converge in 20 iterations. Final step norm: {torch.norm(alpha * step):.4e}, ELBO: {-loss.item():.4f}")
        
    new_prior = torch.tensor([
        q_params[0].item(),
        math.exp(q_params[1].item()),
        q_params[2].item(),
        math.exp(q_params[3].item()),
        q_params[4].item(),
        q_params[5].item()
    ], dtype=torch.float64)
    
    return new_prior

def print_prior(prior_params, label="Prior"):
    mu_m, sigma_m = prior_params[0].item(), prior_params[1].item()
    mu_s, sigma_s = prior_params[2].item(), prior_params[3].item()
    a, b = prior_params[4].item(), prior_params[5].item()
    alpha, beta = math.exp(a), math.exp(b)
    
    # 95% CI for m
    m_lower = mu_m - 1.96 * sigma_m
    m_upper = mu_m + 1.96 * sigma_m
    
    # 95% CI for sqrt(s)
    # log s ~ N(mu_s, sigma_s^2) -> s ~ LogNormal(mu_s, sigma_s^2)
    # The 95% CI for log s is [mu_s - 1.96*sigma_s, mu_s + 1.96*sigma_s]
    # So for sqrt(s) = exp((log s) / 2), the bounds are exponentiated and halved:
    sqrt_s_lower = math.exp((mu_s - 1.96 * sigma_s) / 2)
    sqrt_s_upper = math.exp((mu_s + 1.96 * sigma_s) / 2)
    
    # 95% CI for rho
    from scipy.stats import beta as scipy_beta
    rho_lower = scipy_beta.ppf(0.025, alpha, beta)
    rho_upper = scipy_beta.ppf(0.975, alpha, beta)
    
    print(f"{label}:")
    print(f"  m:       mu={mu_m:7.4f}, sigma={sigma_m:7.4f}      | 95% CI for m:       [{m_lower:7.4f}, {m_upper:7.4f}]")
    print(f"  sqrt(s): mu_s={mu_s:7.4f}, sigma_s={sigma_s:7.4f}    | 95% CI for sqrt(s): [{sqrt_s_lower:7.4f}, {sqrt_s_upper:7.4f}]")
    print(f"  rho:     alpha={alpha:7.2f}, beta={beta:7.2f}        | 95% CI for rho:     [{rho_lower:7.4f}, {rho_upper:7.4f}]")

if __name__ == "__main__":
    current_prior = torch.tensor([
        mu_m_prior, sigma_m_prior,
        mu_s_prior, sigma_s_prior,
        a_prior, b_prior
    ], dtype=torch.float64)

    print_prior(current_prior, label="Initial Prior")
    print("-" * 75)

    for idx, (x, P) in enumerate(dummy_data):
        print(f"Observation {idx+1}: x = {x:.4f}, P = {P:.2f}")
        current_prior = optimize_online(x, P, current_prior)
        
        print_prior(current_prior, label="Updated Prior")
        print("-" * 75)
