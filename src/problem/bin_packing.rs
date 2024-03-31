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
            let duration = (2.9 - input.since.elapsed().as_secs_f64()) * 0.5;

            let (rect0, score0) = step2::divide(input, &dividers, duration);
            let mut input = input.clone();
            input.requests.reverse();
            let (mut rect1, score1) = step2::divide(&input, &dividers, duration);
            rect1.reverse();

            if score0 < score1 {
                rect0
            } else {
                rect1
            }
        } else {
            let mut solver = annealier2d::Annealer2d;
            solver.solve(&input)
        };
        rects
    }
}
