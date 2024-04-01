pub mod annealier2d;
pub mod bin_packing;
pub mod first_fit;
pub mod score_one;

use crate::problem::{Input, Rect};

pub trait Solver {
    fn solve(&self, input: &Input) -> (Vec<Vec<Rect>>, i64);
}
