mod step1;
mod step2;

use super::{Input, Rect};
use crate::solver::Solver;

pub struct BinPacking1d;

impl Solver for BinPacking1d {
    fn solve(&mut self, input: &Input) -> Vec<Vec<Rect>> {
        let dividers = step1::get_best_width(input);
        eprintln!("{:?}", dividers);

        let rects = step2::devide(input, &dividers);
        rects
    }
}
