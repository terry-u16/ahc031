pub mod step1;
mod step2;

use super::{Input, Rect};
use crate::solver::{annealier2d, first_fit::FirstFitPacking, Solver};

pub struct BinPacking1d;

impl Solver for BinPacking1d {
    fn solve(&mut self, input: &Input) -> (Vec<Vec<Rect>>, i64) {
        let (dividers, div_size) = step1::get_best_width(input);
        eprintln!("{:?}", dividers);

        let rects = if div_size >= 3 {
            let (rect0, score0) = step2::divide(input, &dividers);
            let (rect1, score1) = FirstFitPacking::new(0.1).solve(input);

            if score0 < score1 {
                (rect0, score0)
            } else {
                (rect1, score1)
            }
        } else {
            let mut solver = annealier2d::Annealer2d;
            let (rect0, score0) = solver.solve(&input);
            let (rect1, score1) = FirstFitPacking::new(0.1).solve(input);

            if score0 < score1 {
                (rect0, score0)
            } else {
                (rect1, score1)
            }
        };
        rects
    }
}
