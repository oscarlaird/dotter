
import numpy as np
from scipy import stats
from scipy.special import ndtri
import time

# Paste the function here to test it in isolation
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

def check_monotonicity():
    logits = np.linspace(-5, 5, 100)
    xs = 1 / (1 + np.exp(-logits))
    
    target_mu = 2.0
    target_sigma = 1.0
    
    for r in [1, 2]:
        for tail in [0, 1]:
            F_x, slope = compute_analytical_moments(xs, target_mu, target_sigma, r, tail)
            print(f"r={r}, tail={tail}")
            print(f"  F_x range: [{F_x.min():.4g}, {F_x.max():.4g}]")
            print(f"  Slope range: [{slope.min():.4g}, {slope.max():.4g}]")
            
            # Check for NaNs
            if np.isnan(F_x).any() or np.isnan(slope).any():
                print("  NAN DETECTED!")
            
            # Check strict monotonicity of F_x
            diffs = np.diff(F_x)
            if tail == 0:
                # Lower tail integral should increase with x
                if np.any(diffs <= 0):
                    print("  NON-MONOTONIC F_x detected! (should be increasing)")
                    idx = np.where(diffs <= 0)[0]
                    print(f"    Indices: {idx}")
                    print(f"    Values: {F_x[idx]}")
                    print(f"    Next:   {F_x[idx+1]}")
            else:
                # Upper tail integral should decrease with x (since x=P(Y>q) decreases as q increases? No, x=P(Y>q))
                # Wait. xs increases from 0.006 to 0.993.
                # If x = P(Y > q), then as x increases, q decreases.
                # Integral E[Y; Y > q] should increase as x increases (domain gets larger).
                if np.any(diffs <= 0):
                    print("  NON-MONOTONIC F_x detected! (should be increasing)")
                    idx = np.where(diffs <= 0)[0]
                    print(f"    Indices: {idx}")

if __name__ == "__main__":
    check_monotonicity()




