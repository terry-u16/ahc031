mod common;
mod problem;
mod solver;

use crate::problem::{annealier2d::Annealer2d, climbing::Climber, Input};
use problem::{bin_packing::BinPacking1d, first_fit::FirstFitPacking};
use solver::Solver as _;

fn main() {
    let input = Input::read();
    eprintln!("packing_ratio: {:.2}%", input.packing_ratio() * 100.0);

    let mut _solver = Annealer2d;
    let mut _solver = BinPacking1d;
    let mut solver = FirstFitPacking;
    let mut solver = Climber;
    let result = solver.solve(&input);

    for rects in result {
        for rect in rects.iter() {
            println!("{}", rect);
        }
    }
}
