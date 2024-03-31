pub mod step1;
mod step2;

use std::time::Instant;

use super::{Input, Rect};
use crate::solver::Solver;

pub struct BinPacking1d {
    duration: f64,
}

impl BinPacking1d {
    pub fn new(duration: f64) -> Self {
        Self { duration }
    }
}

impl Solver for BinPacking1d {
    fn solve(&self, input: &Input) -> (Vec<Vec<Rect>>, i64) {
        let since = Instant::now();
        let (dividers, div_size) = step1::get_best_width(input);

        if div_size >= 3 {
            let duration = self.duration - since.elapsed().as_secs_f64();
            step2::divide(input, &dividers, duration)
        } else {
            (vec![], i64::MAX)
        }
    }
}
