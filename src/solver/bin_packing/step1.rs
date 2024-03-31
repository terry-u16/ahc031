use itertools::Itertools as _;
use rand::{Rng as _, SeedableRng as _};

use crate::{common::ChangeMinMax as _, problem::Input};

pub fn get_best_width(input: &Input) -> (Vec<Vec<i32>>, usize) {
    let mut dividers = vec![0, Input::W];
    let mut div_size = 1;

    for div in 2..input.n {
        let state = gen_dividers(input, div);
        let score = state.calc_score(input).unwrap();

        if score > 0 {
            break;
        }

        let mut divs = vec![0];

        for &w in state.widths.iter() {
            let w = w + divs.last().unwrap();
            divs.push(w);
        }

        dividers = divs;
        div_size = div;
    }

    let mut divider_set = vec![dividers];

    if div_size == 1 {
        return (divider_set, div_size);
    }

    for _ in 0..9 {
        let state = gen_dividers(input, div_size);
        let score = state.calc_score(input).unwrap();

        if score > 0 {
            continue;
        }

        let mut divs = vec![0];

        for &w in state.widths.iter() {
            let w = w + divs.last().unwrap();
            divs.push(w);
        }

        divider_set.push(divs);
    }

    (divider_set, div_size)
}

fn gen_dividers(input: &Input, div: usize) -> State {
    let lines = (0..=div)
        .map(|i| (Input::W as usize * i / div) as i32)
        .collect_vec();
    let widths = lines
        .iter()
        .tuple_windows()
        .map(|(&a, &b)| b - a)
        .collect_vec();
    let mut state = State::new(widths);
    state = annealing(input, state, 0.01);

    state.widths.sort_unstable();
    state
}

#[derive(Debug, Clone)]
struct State {
    widths: Vec<i32>,
}

impl State {
    fn new(widths: Vec<i32>) -> Self {
        Self { widths }
    }

    fn calc_score(&self, input: &Input) -> Result<i64, ()> {
        if self.widths.iter().any(|&w| w <= 0) {
            return Err(());
        }

        let mut score = 0;

        let mut used = vec![false; self.widths.len()];
        let bins = self.widths.iter().map(|&w| w * Input::W).collect_vec();

        for reqs in input.requests.iter() {
            let mut bins = bins.clone();
            used.fill(false);

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
                used[best_i] = true;
            }

            for &b in bins.iter() {
                score += (-b).max(0) as i64;
            }

            // 未使用の仕切りがあればペナルティ
            score += used.iter().filter(|&&b| !b).count() as i64 * 100;
        }

        // 最大幅が小さいほどボーナス
        score -= (Input::W / 2 - *self.widths.iter().max().unwrap()) as i64 / 10;

        Ok(score)
    }
}

fn annealing(input: &Input, initial_solution: State, duration: f64) -> State {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_score(input).unwrap();
    let mut best_score = current_score;

    let mut all_iter = 0;
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();

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
        let i = rng.gen_range(0..solution.widths.len());
        let mut j = rng.gen_range(0..solution.widths.len() - 1);

        if j >= i {
            j += 1;
        }

        let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
        let dx = sign * 10f64.powf(rng.gen_range(0.0..2.0)).round() as i32;
        let mut new_state = solution.clone();
        new_state.widths[i] -= dx;
        new_state.widths[j] += dx;

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
