mod width_divide;

use super::{Input, Rect};
use crate::solver::Solver;

pub struct BinPacking1d;

impl Solver for BinPacking1d {
    fn solve(&mut self, input: &Input) -> Vec<Vec<Rect>> {
        let dividers = width_divide::get_best_width(input);
        eprintln!("{:?}", dividers);

        todo!()
    }
}
