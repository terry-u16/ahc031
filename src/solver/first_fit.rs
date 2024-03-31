mod step1;
mod step2;

use super::{Input, Rect};
use crate::solver::Solver;

pub struct FirstFitPacking {
    duration1: f64,
    duration2: f64,
}

impl FirstFitPacking {
    pub fn new(duration1: f64, duration2: f64) -> Self {
        Self {
            duration1,
            duration2,
        }
    }
}

impl Solver for FirstFitPacking {
    fn solve(&self, input: &Input) -> (Vec<Vec<Rect>>, i64) {
        let (dividers, separators) = step1::generate_init_solution(input, self.duration1);
        step2::devide(input, &dividers, separators, self.duration2)
    }
}
