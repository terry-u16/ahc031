mod step1;
mod step2;

use rand::{Rng as _, SeedableRng as _};

use super::{Input, Rect};
use crate::{common::ChangeMinMax, solver::Solver};

pub struct FirstFitPacking {
    duration: f64,
}

impl FirstFitPacking {
    pub fn new(duration: f64) -> Self {
        Self { duration }
    }
}

impl Solver for FirstFitPacking {
    fn solve(&mut self, input: &Input) -> (Vec<Vec<Rect>>, i64) {
        let (dividers, separators) = step1::generate_init_solution(input, 0.1);
        step2::devide(input, &dividers, separators)
    }
}
