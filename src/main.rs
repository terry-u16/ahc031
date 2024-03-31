mod common;
mod problem;
mod solver;

use crate::problem::Input;
use solver::bin_packing::BinPacking1d;
use solver::Solver as _;

fn main() {
    let input = Input::read();
    eprintln!("packing_ratio: {:.2}%", input.packing_ratio * 100.0);

    let mut solver = BinPacking1d;
    let result = solver.solve(&input);

    for rects in result {
        for rect in rects.iter() {
            println!("{}", rect);
        }
    }
}
