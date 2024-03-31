pub mod annealier2d;
pub mod bin_packing;
pub mod first_fit;
pub mod break_and_best_fit;

use std::{fmt::Display, time::Instant};

use proconio::input;

#[derive(Debug, Clone)]
pub struct Input {
    pub days: usize,
    pub n: usize,
    pub requests: Vec<Vec<i32>>,
    pub since: Instant,
}

impl Input {
    pub const W: i32 = 1000;

    pub fn read() -> Self {
        input! {
            _w: i32,
            days: usize,
            n: usize,
            requests: [[i32; n]; days],
        }

        let since = Instant::now();

        Self {
            days,
            n,
            requests,
            since,
        }
    }

    pub fn packing_ratio(&self) -> f64 {
        let mut sum = self
            .requests
            .iter()
            .map(|reqs| reqs.iter().sum::<i32>() as f64)
            .sum::<f64>();
        sum /= self.days as f64 * Self::W as f64 * Self::W as f64;
        sum
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    pub fn is_valid(&self) -> bool {
        self.x0 < self.x1 && self.y0 < self.y1
    }

    pub fn area(&self) -> i32 {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

impl Display for Rect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {} {}", self.y0, self.x0, self.y1, self.x1)
    }
}
