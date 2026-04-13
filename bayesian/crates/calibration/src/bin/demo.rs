use statrs::distribution::{Beta, ContinuousCDF};
use calibration::{optimize_online, VariationalParams};

fn print_prior(prior_params: &VariationalParams, label: &str) {
    let mu_m = prior_params.mu_m;
    let sigma_m = prior_params.sigma_m;
    let mu_s = prior_params.mu_s;
    let sigma_s = prior_params.sigma_s;
    let alpha = prior_params.alpha();
    let beta = prior_params.beta();
    
    let m_lower = mu_m - 1.96 * sigma_m;
    let m_upper = mu_m + 1.96 * sigma_m;
    
    let sqrt_s_lower = ((mu_s - 1.96 * sigma_s) / 2.0).exp();
    let sqrt_s_upper = ((mu_s + 1.96 * sigma_s) / 2.0).exp();
    
    let beta_dist = Beta::new(alpha, beta).unwrap();
    let rho_lower = beta_dist.inverse_cdf(0.025);
    let rho_upper = beta_dist.inverse_cdf(0.975);
    
    println!("{}:", label);
    println!("  m:       mu={:7.4}, sigma={:7.4}      | 95% CI for m:       [{:7.4}, {:7.4}]", mu_m, sigma_m, m_lower, m_upper);
    println!("  sqrt(s): mu_s={:7.4}, sigma_s={:7.4}    | 95% CI for sqrt(s): [{:7.4}, {:7.4}]", mu_s, sigma_s, sqrt_s_lower, sqrt_s_upper);
    println!("  rho:     alpha={:7.2}, beta={:7.2}        | 95% CI for rho:     [{:7.4}, {:7.4}]", alpha, beta, rho_lower, rho_upper);
}

fn main() {
    let dummy_data = [
        (0.15, 1.0),
        (0.19, 1.0),
        (0.10, 1.0),
        (0.18, 1.0),
        (0.10, 1.2),
        (0.15, 1.2),
    ];
    
    let mut current_prior = VariationalParams::default_calibration();
    
    print_prior(&current_prior, "Initial Prior");
    println!("{:-<75}", "");
    
    for (idx, (x, p_val)) in dummy_data.iter().enumerate() {
        println!("Observation {}: x = {:.4}, P = {:.2}", idx + 1, x, p_val);
        current_prior = optimize_online(*x, *p_val, &current_prior, false);
        
        print_prior(&current_prior, "Updated Prior");
        println!("{:-<75}", "");
    }
}
