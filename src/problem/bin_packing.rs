pub mod step1;
mod step2;

use super::{Input, Rect};
use crate::{problem::annealier2d, solver::Solver};

pub struct BinPacking1d;

impl Solver for BinPacking1d {
    fn solve(&mut self, input: &Input) -> Vec<Vec<Rect>> {
        let (dividers, div_size) = step1::get_best_width(input);
        eprintln!("{:?}", dividers);

        let rects = if div_size >= 3 {
            step2::divide(input, &dividers)
        } else {
            let mut solver = annealier2d::Annealer2d;
            solver.solve(&input)
        };
        rects
    }
}
