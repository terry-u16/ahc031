mod common;
mod problem;
mod solver;

use crate::problem::{annealier2d::Annealer2d, break_and_best_fit::BreakAndBestFit, Input};
use problem::{bin_packing::BinPacking1d, first_fit::FirstFitPacking};
use solver::Solver as _;

fn main() {
    let input = Input::read();
    eprintln!("packing_ratio: {:.2}%", input.packing_ratio * 100.0);

    let mut _solver = Annealer2d;
    let mut solver = FirstFitPacking;
    let mut solver = BinPacking1d;
    let mut solver = BreakAndBestFit;
    let result = solver.solve(&input);

    for rects in result {
        for rect in rects.iter() {
            println!("{}", rect);
        }
    }
}
