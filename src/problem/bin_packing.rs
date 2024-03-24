use itertools::Itertools;
use rand::Rng as _;

use crate::{common::ChangeMinMax, solver::Solver};

use super::{Input, Rect};

pub struct BinPacking1d;

impl Solver for BinPacking1d {
    fn solve(&mut self, input: &Input) -> Vec<Vec<Rect>> {
        let sum_reqs = input
            .requests
            .iter()
            .map(|reqs| reqs.iter().map(|&r| r as i64).sum::<i64>())
            .sum::<i64>();
        let average = sum_reqs as f64 / input.days as f64 / (Input::W * Input::W) as f64 * 100.0;
        eprintln!(
            "D = {}, N = {}, average: {:.2}%",
            input.days, input.n, average
        );

        for div in 2..=input.n {
            let lines = (0..=div)
                .map(|i| (Input::W as usize * i / div) as i32)
                .collect_vec();
            let state = State::new(lines);
            let state = annealing(input, state, 0.1);
            let mut w = state
                .lines
                .iter()
                .tuple_windows()
                .map(|(&a, &b)| b - a)
                .collect_vec();
            w.sort_unstable();
            let score = state.calc_score(input).unwrap();
            eprintln!("[Div {}] score: {}, {:?}", div, score, w);

            if score > 0 {
                break;
            }
        }

        todo!()
    }
}

#[derive(Debug, Clone)]
struct State {
    lines: Vec<i32>,
}

impl State {
    fn new(lines: Vec<i32>) -> Self {
        Self { lines }
    }

    fn calc_score(&self, input: &Input) -> Result<i64, ()> {
        let mut bins = vec![];

        for (&a, &b) in self.lines.iter().tuple_windows() {
            let diff = b - a;

            if diff < 10 {
                return Err(());
            }

            bins.push(diff * Input::W);
        }

        let mut score = 0;

        for reqs in input.requests.iter() {
            let mut bins = bins.clone();

            // best-fit algoithm
            for &req in reqs.iter().rev() {
                let mut best_i = !0;
                let mut best_score = i64::MAX;

                for (i, &bin) in bins.iter().enumerate() {
                    let score = (bin - req) as i64;
                    let score = if score < 0 {
                        score * -1000000000
                    } else {
                        score
                    };

                    if best_score.change_min(score) {
                        best_i = i;
                    }
                }

                bins[best_i] -= req;
            }

            for &b in bins.iter() {
                score += (-b).max(0) as i64;
            }
        }

        Ok(score)
    }
}

fn annealing(input: &Input, initial_solution: State, duration: f64) -> State {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_score(input).unwrap();
    let mut best_score = current_score;

    let mut all_iter = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e6 * input.days as f64;
    let temp1 = 1e0;
    let mut inv_temp = 1.0 / temp0;

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);
            inv_temp = 1.0 / temp;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let i = rng.gen_range(1..solution.lines.len() - 1);
        let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
        let dx = sign * 10f64.powf(rng.gen_range(0.0..2.0)).round() as i32;
        let mut new_state = solution.clone();
        new_state.lines[i] += dx;

        // スコア計算
        let Ok(new_score) = new_state.calc_score(input) else {
            continue;
        };
        let score_diff = new_score - current_score;

        if score_diff <= 0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            solution = new_state;

            if best_score.change_min(current_score) {
                best_solution = solution.clone();
            }
        }
    }

    best_solution
}
