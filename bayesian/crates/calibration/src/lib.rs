use num_dual::*;
use nalgebra::{Const, OVector, SVector};
use std::f64;

trait CustomFuncs {
    fn my_lgamma(&self) -> Self;
    fn my_digamma(&self) -> Self;
}

impl CustomFuncs for Dual2SVec64<6> {
    fn my_lgamma(&self) -> Self {
        let f0 = statrs::function::gamma::ln_gamma(self.re);
        let f1 = statrs::function::gamma::digamma(self.re);
        let f2 = polygamma::polygamma(1, self.re).unwrap();
        Dual2SVec64::new(
            f0,
            &self.v1 * f1,
            &self.v2 * f1 + self.v1.tr_mul(&self.v1) * f2,
        )
    }

    fn my_digamma(&self) -> Self {
        let f0 = statrs::function::gamma::digamma(self.re);
        let f1 = polygamma::polygamma(1, self.re).unwrap();
        let f2 = polygamma::polygamma(2, self.re).unwrap();
        Dual2SVec64::new(
            f0,
            &self.v1 * f1,
            &self.v2 * f1 + self.v1.tr_mul(&self.v1) * f2,
        )
    }
}

fn logsumexp(slice: &[Dual2SVec64<6>]) -> Dual2SVec64<6> {
    let mut max_val = f64::NEG_INFINITY;
    for x in slice {
        if x.re > max_val {
            max_val = x.re;
        }
    }
    let max_val_dual = Dual2SVec64::from_re(max_val);
    let mut sum_exp = Dual2SVec64::from_re(0.0);
    for x in slice {
        sum_exp += (*x - max_val_dual).exp();
    }
    sum_exp.ln() + max_val_dual
}

fn evaluate_j(params: OVector<Dual2SVec64<6>, Const<6>>, x: f64, p_val: f64, prior_params: &[f64; 6]) -> Dual2SVec64<6> {
    let mu_m_q = params[0];
    let sigma_m_q = params[1].exp();
    let mu_s_q = params[2];
    let sigma_s_q = params[3].exp();
    let a_q = params[4];
    let b_q = params[5];
    
    let alpha_q = a_q.exp();
    let beta_q = b_q.exp();
    
    let mu_m_p = prior_params[0];
    let sigma_m_p = prior_params[1];
    let mu_s_p = prior_params[2];
    let sigma_s_p = prior_params[3];
    let alpha_p = prior_params[4].exp();
    let beta_p = prior_params[5].exp();

    let e_log_rho = alpha_q.my_digamma() - (alpha_q + beta_q).my_digamma();
    let e_log_1_minus_rho = beta_q.my_digamma() - (alpha_q + beta_q).my_digamma();
    
    let mut terms = Vec::new();
    let term1 = e_log_rho - Dual2SVec64::from_re(p_val.ln());
    terms.push(term1);
    
    for k in [-1, 0, 1] {
        let k_f = k as f64;
        let p_dual = Dual2SVec64::from_re(p_val);
        let x_dual = Dual2SVec64::from_re(x);
        let pi_term = Dual2SVec64::from_re(0.5 * (2.0 * f64::consts::PI).ln());
        
        let mut term = e_log_1_minus_rho - pi_term - mu_s_q * Dual2SVec64::from_re(0.5);
        let exponent = -mu_s_q + sigma_s_q * sigma_s_q * Dual2SVec64::from_re(0.5);
        let factor = exponent.exp() * Dual2SVec64::from_re(0.5);
        let squared_diff = (x_dual - mu_m_q - p_dual * Dual2SVec64::from_re(k_f)) * (x_dual - mu_m_q - p_dual * Dual2SVec64::from_re(k_f));
        let val = factor * (squared_diff + sigma_m_q * sigma_m_q);
        term = term - val;
        terms.push(term);
    }
    
    let likelihood_bound = logsumexp(&terms);
    
    let half = Dual2SVec64::from_re(0.5);
    
    let sigma_m_p_dual = Dual2SVec64::from_re(sigma_m_p);
    let mu_m_p_dual = Dual2SVec64::from_re(mu_m_p);
    let kl_m = (sigma_m_p_dual / sigma_m_q).ln() 
             + (sigma_m_q * sigma_m_q + (mu_m_q - mu_m_p_dual) * (mu_m_q - mu_m_p_dual)) 
               / (sigma_m_p_dual * sigma_m_p_dual * Dual2SVec64::from_re(2.0)) 
             - half;
             
    let sigma_s_p_dual = Dual2SVec64::from_re(sigma_s_p);
    let mu_s_p_dual = Dual2SVec64::from_re(mu_s_p);
    let kl_s = (sigma_s_p_dual / sigma_s_q).ln() 
             + (sigma_s_q * sigma_s_q + (mu_s_q - mu_s_p_dual) * (mu_s_q - mu_s_p_dual)) 
               / (sigma_s_p_dual * sigma_s_p_dual * Dual2SVec64::from_re(2.0)) 
             - half;
             
    let log_b_p = statrs::function::gamma::ln_gamma(alpha_p) + statrs::function::gamma::ln_gamma(beta_p) - statrs::function::gamma::ln_gamma(alpha_p + beta_p);
    let log_b_p_dual = Dual2SVec64::from_re(log_b_p);
    
    let log_b_q = alpha_q.my_lgamma() + beta_q.my_lgamma() - (alpha_q + beta_q).my_lgamma();
    
    let alpha_p_dual = Dual2SVec64::from_re(alpha_p);
    let beta_p_dual = Dual2SVec64::from_re(beta_p);
    
    let kl_beta = log_b_p_dual - log_b_q 
                + (alpha_q - alpha_p_dual) * alpha_q.my_digamma()
                + (beta_q - beta_p_dual) * beta_q.my_digamma()
                - (alpha_q + beta_q - alpha_p_dual - beta_p_dual) * (alpha_q + beta_q).my_digamma();
                
    likelihood_bound - kl_m - kl_s - kl_beta
}

pub fn optimize_online(x: f64, p_val: f64, prior_params: &[f64; 6]) -> [f64; 6] {
    let mut q_params = SVector::<f64, 6>::new(
        prior_params[0],
        prior_params[1].ln(),
        prior_params[2],
        prior_params[3].ln(),
        prior_params[4],
        prior_params[5]
    );

    #[cfg(not(target_arch = "wasm32"))]
    let mut total_duration = std::time::Duration::new(0, 0);

    for i in 0..20 {
        #[cfg(not(target_arch = "wasm32"))]
        let start = std::time::Instant::now();
        let (loss, grad, mut hessian) = num_dual::hessian(
            |p| -evaluate_j(p, x, p_val, prior_params),
            q_params,
        );
        #[cfg(not(target_arch = "wasm32"))]
        {
            total_duration += start.elapsed();
        }


        let mut eigen = hessian.symmetric_eigen();
        for ev in eigen.eigenvalues.iter_mut() {
            if *ev < 1e-4 {
                *ev = 1e-4;
            }
        }
        let hessian_pd = eigen.eigenvectors * nalgebra::Matrix6::from_diagonal(&eigen.eigenvalues) * eigen.eigenvectors.transpose();
        
        let step = -hessian_pd.lu().solve(&grad).unwrap();
        
        let mut alpha = 1.0;
        let c = 1e-4;
        let mut new_params = q_params;
        let mut new_loss = 0.0;
        
        while alpha > 1e-6 {
            new_params = q_params + alpha * step;
            let (nl, _, _) = num_dual::hessian(|p| -evaluate_j(p, x, p_val, prior_params), new_params);
            new_loss = nl;
            if new_loss <= loss + c * alpha * grad.dot(&step) {
                break;
            }
            alpha *= 0.5;
        }
        
        q_params = new_params;
        
        if (alpha * step).norm() < 1e-6 {
            #[cfg(not(target_arch = "wasm32"))]
            {
            let avg_duration = total_duration / (i + 1) as u32;
            println!("Converged in {} iterations. ELBO: {:.4} (Time per iter: {:?})", i + 1, -loss, avg_duration);
            }
            break;
        }
        
        if i == 19 {
            #[cfg(not(target_arch = "wasm32"))]
            {
            let avg_duration = total_duration / 20;
            println!("Did not converge in 20 iterations. Final step norm: {:.4e}, ELBO: {:.4} (Time per iter: {:?})", (alpha * step).norm(), -loss, avg_duration);
            }
        }
    }
    
    [
        q_params[0],
        q_params[1].exp(),
        q_params[2],
        q_params[3].exp(),
        q_params[4],
        q_params[5],
    ]
}

