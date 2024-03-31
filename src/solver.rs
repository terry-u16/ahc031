pub mod annealier2d;
pub mod bin_packing;
pub mod first_fit;

use crate::problem::{Input, Rect};

pub trait Solver {
    fn solve(&self, input: &Input) -> (Vec<Vec<Rect>>, i64);
}
