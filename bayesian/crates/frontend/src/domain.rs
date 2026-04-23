use std::time::{SystemTime, UNIX_EPOCH};

use bayesian::calibration::VariationalParams;
use layout::{LikelihoodModel, ROOT_SYMBOL, SPACE_SYMBOL, STOP_SYMBOL};
use statrs::distribution::{Beta, ContinuousCDF};

pub const DEFAULT_PERIOD: f64 = 1.1;

#[derive(Clone, Debug, PartialEq)]
pub struct AutoCalibrationState {
    pub mu_delay: bool,
    pub stddev_delay: bool,
    pub outliers: bool,
}

impl Default for AutoCalibrationState {
    fn default() -> Self {
        Self {
            mu_delay: true,
            stddev_delay: true,
            outliers: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LikelihoodIntervals {
    pub mu_delay: (f64, f64),
    pub stddev_delay: (f64, f64),
    pub outliers: (f64, f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedLikelihoodModel {
    pub model: LikelihoodModel,
    pub intervals: Option<LikelihoodIntervals>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorMode {
    Light,
    Dark,
}

impl ColorMode {
    pub fn toggle(self) -> Self {
        match self {
            Self::Light => Self::Dark,
            Self::Dark => Self::Light,
        }
    }

    pub fn is_light(self) -> bool {
        matches!(self, Self::Light)
    }
}

pub fn default_likelihood_model() -> LikelihoodModel {
    LikelihoodModel {
        mu_delay: 0.0,
        stddev_delay: 0.064,
        outliers: 0.08,
        period: DEFAULT_PERIOD,
    }
}

pub fn predictive_stddev(mu_s: f64, sigma_s: f64, sigma_m: f64) -> f64 {
    (f64::exp(mu_s + sigma_s.powi(2) / 2.0) + sigma_m.powi(2)).sqrt()
}

pub fn variational_params_to_likelihood_model(
    params: VariationalParams,
    period: f64,
) -> CalibratedLikelihoodModel {
    let alpha = params.log_alpha.exp();
    let beta = params.log_beta.exp();
    let beta_dist = Beta::new(alpha, beta).expect("valid beta calibration distribution");
    CalibratedLikelihoodModel {
        model: LikelihoodModel {
            mu_delay: params.mu_m,
            stddev_delay: predictive_stddev(params.mu_s, params.sigma_s, params.sigma_m),
            outliers: beta_dist.inverse_cdf(0.5),
            period,
        },
        intervals: Some(LikelihoodIntervals {
            mu_delay: (
                params.mu_m - 1.96 * params.sigma_m,
                params.mu_m + 1.96 * params.sigma_m,
            ),
            stddev_delay: (
                predictive_stddev(
                    params.mu_s - 1.96 * params.sigma_s,
                    params.sigma_s,
                    params.sigma_m,
                ),
                predictive_stddev(
                    params.mu_s + 1.96 * params.sigma_s,
                    params.sigma_s,
                    params.sigma_m,
                ),
            ),
            outliers: (beta_dist.inverse_cdf(0.025), beta_dist.inverse_cdf(0.975)),
        }),
    }
}

const N_SKIP_PRACTICE_PHRASES: usize = 6;
const PRACTICE_PHRASES_TEXT: &str =
    include_str!("../../../../frontend/src/pages/practice-phrases.txt");

fn format_practice_phrase(phrase: &str) -> String {
    format!(" {phrase}Z")
}

pub fn practice_phrases() -> Vec<String> {
    PRACTICE_PHRASES_TEXT
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(format_practice_phrase)
        .collect()
}

pub fn random_practice_phrase(excluding: Option<&str>) -> String {
    let phrases = practice_phrases();
    let source = if phrases.len() > N_SKIP_PRACTICE_PHRASES {
        &phrases[N_SKIP_PRACTICE_PHRASES..]
    } else {
        &phrases[..]
    };
    let source = if source.is_empty() {
        &phrases[..]
    } else {
        source
    };
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current time after unix epoch")
        .subsec_nanos() as usize;
    let mut index = nanos % source.len().max(1);
    if let Some(excluding) = excluding {
        if source.len() > 1 && source[index] == excluding {
            index = (index + 1) % source.len();
        }
    }
    source[index].clone()
}

pub fn color_from_letter(letter: char) -> (u8, u8, u8) {
    match letter {
        'a' | 'j' | 'r' | 'Z' | '$' | '.' | ',' | '\'' => (255, 77, 77),
        'b' | 'n' | 'w' => (77, 121, 255),
        'c' | 'i' | 'p' | 'v' => (153, 77, 255),
        'd' | 'k' | 't' => (204, 153, 102),
        'e' | 'm' | 'u' => (160, 160, 160),
        'f' | 'l' | 'q' | 'y' => (255, 225, 77),
        'g' | 's' | 'z' => (34, 177, 76),
        'h' | 'o' | 'x' => (255, 153, 51),
        ' ' => (255, 255, 255),
        'N' => (0, 168, 150),
        'Q' => (139, 92, 246),
        'U' => (245, 158, 11),
        _ => (255, 255, 255),
    }
}

pub fn display_text(symbol: char) -> String {
    match symbol {
        SPACE_SYMBOL => " ".to_string(),
        'N' => "#".to_string(),
        'Q' => "SYM".to_string(),
        'U' => "SHIFT".to_string(),
        STOP_SYMBOL => String::new(),
        _ => symbol.to_string(),
    }
}

fn is_ascii_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn is_lowercase_letter(c: char) -> bool {
    c.is_ascii_lowercase()
}

pub fn trie_wire_to_display_prefix(s: &str) -> String {
    let chars = s.chars().collect::<Vec<_>>();
    let mut i = 0usize;
    let mut out = String::new();
    while i < chars.len() {
        let c = chars[i];
        match c {
            STOP_SYMBOL => {
                i += 1;
            }
            SPACE_SYMBOL => {
                out.push(' ');
                i += 1;
            }
            'N' => {
                if i + 1 < chars.len() && is_ascii_digit(chars[i + 1]) {
                    out.push(chars[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            'Q' => {
                if i + 1 < chars.len() {
                    out.push(chars[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            'U' => {
                if i + 1 < chars.len() && is_lowercase_letter(chars[i + 1]) {
                    out.push(chars[i + 1].to_ascii_uppercase());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

pub fn offscreen_prefix_text(scroll_root: &str) -> String {
    let root = ROOT_SYMBOL.to_string();
    if scroll_root == root {
        String::new()
    } else {
        trie_wire_to_display_prefix(&scroll_root[1..])
    }
}
