mod common;
mod problem;
mod solver;

use crate::problem::{annealier2d::Annealer2d, Input};
use problem::{bin_packing::BinPacking1d, first_fit::FirstFitPacking};
use solver::Solver as _;

fn main() {
    let input = Input::read();
    let mut _solver = Annealer2d;
    let mut _solver = BinPacking1d;
    let mut solver = FirstFitPacking;
    let result = solver.solve(&input);

    for rects in result {
        for rect in rects.iter() {
            println!("{}", rect);
        }
    }
}
