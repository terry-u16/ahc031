use crate::problem::{Input, Rect};

pub trait Solver {
    fn solve(&mut self, input: &Input) -> Vec<Vec<Rect>>;
}
