use std::{fmt::Display, time::Instant};

use proconio::input;

use crate::params::ParamSuggester;

#[derive(Debug, Clone)]
pub struct Input {
    pub days: usize,
    pub n: usize,
    pub requests: Vec<Vec<i32>>,
    pub packing_ratio: f64,
    pub since: Instant,
    pub first_fit_config: FirstFitConfig,
}

impl Input {
    pub const W: i32 = 1000;

    pub fn read() -> Self {
        input! {
            _w: i32,
            days: usize,
            n: usize,
            requests: [[i32; n]; days],
        }

        let mut packing_ratio = requests
            .iter()
            .map(|reqs| reqs.iter().sum::<i32>() as f64)
            .sum::<f64>();
        packing_ratio /= days as f64 * Self::W as f64 * Self::W as f64;

        let e = (1.0 - packing_ratio) * (Input::W * Input::W) as f64;

        let first_fit_config = FirstFitConfig::new(days, n, e);

        let since = Instant::now();

        Self {
            days,
            n,
            requests,
            packing_ratio,
            first_fit_config,
            since,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    pub fn is_valid(&self) -> bool {
        self.x0 < self.x1 && self.y0 < self.y1
    }

    pub fn area(&self) -> i32 {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

impl Display for Rect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {} {}", self.y0, self.x0, self.y1, self.x1)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FirstFitConfig {
    pub step1_ratio: f64,
    pub step1_temp0: f64,
    pub step1_temp1: f64,
    pub step2_temp0: f64,
    pub step2_temp1: f64,
}

impl FirstFitConfig {
    fn new(d: usize, n: usize, e: f64) -> Self {
        //let args = std::env::args().collect_vec();
        //let step1_ratio = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.1);
        //let step1_temp0 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1e7);
        //let step1_temp1 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1e0);
        //let step2_temp0 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2e2);
        //let step2_temp1 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3e0);

        let step1_ratio = ParamSuggester::gen_ratio_pred().suggest(d, n, e);
        let step1_temp0 = ParamSuggester::gen_t00_pred().suggest(d, n, e);
        let step1_temp1 = ParamSuggester::gen_t01_pred().suggest(d, n, e);
        let step2_temp0 = ParamSuggester::gen_t10_pred().suggest(d, n, e);
        let step2_temp1 = ParamSuggester::gen_t11_pred().suggest(d, n, e);

        Self {
            step1_ratio,
            step1_temp0,
            step1_temp1,
            step2_temp0,
            step2_temp1,
        }
    }
}
