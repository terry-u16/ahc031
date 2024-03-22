#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Input {
    days: usize,
    n: usize,
    requests: Vec<Vec<i64>>,
}

impl Input {
    const W: i64 = 1000;

    fn read() -> Self {
        input! {
            _w: i64,
            days: usize,
            n: usize,
            requests: [[i64; n]; days],
        }

        Self { days, n, requests }
    }
}

fn main() {
    let input = Input::read();

    for _ in 0..input.days {
        for i in 0..input.n {
            let x0 = Input::W * i as i64 / input.n as i64;
            let x1 = Input::W * (i + 1) as i64 / input.n as i64;
            println!("{} {} {} {}", x0, 0, x1, Input::W);
        }
    }
}
