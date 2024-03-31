mod common;
mod problem;
mod solver;

use crate::common::ChangeMinMax;
use crate::problem::Input;
use crate::solver::annealier2d::Annealer2d;
use crate::solver::first_fit::FirstFitPacking;
use solver::bin_packing::BinPacking1d;
use solver::Solver as _;

fn main() {
    let input = Input::read();
    eprintln!("packing_ratio: {:.2}%", input.packing_ratio * 100.0);

    let first_fit = FirstFitPacking::new(0.3, 2.0);
    let (mut best_result, mut best_score) = first_fit.solve(&input);

    eprintln!("first_fit score: {}", best_score);

    let duration = 2.8 - input.since.elapsed().as_secs_f64();
    let bin_packing = BinPacking1d::new(duration);
    let (result, score) = bin_packing.solve(&input);

    eprintln!("bin_packing score: {}", score);

    if best_score.change_min(score) {
        best_result = result;
    }

    let duration = 2.98 - input.since.elapsed().as_secs_f64();

    if duration > 0.0 {
        let annealer2d = Annealer2d::new(duration);
        let (result, score) = annealer2d.solve(&input);
        eprintln!("annealer2d score: {}", score);

        if best_score.change_min(score) {
            best_result = result;
        }
    }

    eprintln!("score: {}", best_score);

    for rects in best_result {
        for rect in rects.iter() {
            println!("{}", rect);
        }
    }
}
