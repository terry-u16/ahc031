use rand::{Rng as _, SeedableRng as _};

use super::{Input, Rect};
use crate::{common::ChangeMinMax, solver::Solver};

pub struct FirstFitPacking;

impl Solver for FirstFitPacking {
    fn solve(&mut self, input: &Input) -> Vec<Vec<Rect>> {
        let state = State::new(vec![Input::W]);
        let state = annealing(input, state, 2.9);
        eprintln!("{:?}", state);

        state.to_rects(input)
    }
}

#[derive(Debug, Clone)]
struct State {
    widths: Vec<i32>,
}

impl State {
    fn new(widths: Vec<i32>) -> Self {
        Self { widths }
    }

    fn len(&self) -> usize {
        self.widths.len()
    }

    fn calc_score(&self, input: &Input) -> i64 {
        let mut widths = self.widths.clone();
        widths.sort();

        let mut y_lanes = vec![0; widths.len()];

        let mut score = 1;

        for (day, requests) in input.requests.iter().enumerate() {
            let line_mul = if day == 0 || day == input.days - 1 {
                1
            } else {
                2
            };

            y_lanes.fill(0);

            for &request in requests.iter().rev() {
                let mut next_i = !0;
                let mut next_y = i32::MAX;

                // first-fitをする
                for (i, (&width, &y)) in widths.iter().zip(y_lanes.iter()).enumerate() {
                    let dy = (request + width - 1) / width;
                    let y = y + dy;

                    if next_y.change_min(y) {
                        next_i = i;
                    }

                    if next_y <= Input::W {
                        break;
                    }
                }

                // 各レーン2個目以降はコストがかかる
                if y_lanes[next_i] > 0 {
                    score += widths[next_i] as i64 * line_mul;
                }

                // 溢れるケース
                score +=
                    100 * ((next_y - Input::W.max(y_lanes[next_i])).max(0) * widths[next_i]) as i64;

                y_lanes[next_i] = next_y;
            }
        }

        score
    }

    fn to_rects(&self, input: &Input) -> Vec<Vec<Rect>> {
        let mut all_rects = vec![];
        let mut widths = self.widths.clone();
        widths.sort();

        let mut y_lanes = vec![0; widths.len()];

        let mut score = 1;

        for (day, requests) in input.requests.iter().enumerate() {
            let line_mul = if day == 0 || day == input.days - 1 {
                1
            } else {
                2
            };

            let mut lane_rects = vec![vec![]; widths.len()];
            y_lanes.fill(0);

            for (rect_j, &request) in requests.iter().enumerate().rev() {
                let mut next_i = !0;
                let mut next_y = i32::MAX;

                // first-fitをする
                for (i, (&width, &y)) in widths.iter().zip(y_lanes.iter()).enumerate() {
                    let dy = (request + width - 1) / width;
                    let y = y + dy;

                    if next_y.change_min(y) {
                        next_i = i;
                    }

                    if next_y <= Input::W {
                        break;
                    }
                }

                // 各レーン2個目以降はコストがかかる
                if y_lanes[next_i] > 0 {
                    score += widths[next_i] as i64 * line_mul;
                }

                // 溢れるケース
                score +=
                    100 * ((next_y - Input::W.max(y_lanes[next_i])).max(0) * widths[next_i]) as i64;

                lane_rects[next_i].push(next_y - y_lanes[next_i]);
                y_lanes[next_i] = next_y;
            }

            let mut rects = vec![];
            let mut x = 0;

            for i in 0..widths.len() {
                let mut y = 0;

                for &dy in lane_rects[i].iter() {
                    rects.push(Rect::new(x, y, x + widths[i], y + dy));
                    y += dy;
                }

                if let Some(rect) = rects.last_mut() {
                    rect.y1 = Input::W;
                }

                x += widths[i];
            }

            glidesort::sort_by_key(&mut rects, |r| r.area());
            all_rects.push(rects);
        }

        all_rects
    }
}

fn annealing(input: &Input, initial_solution: State, duration: f64) -> State {
    let mut state = initial_solution;
    let mut best_state = state.clone();
    let mut current_score = state.calc_score(input);
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e8;
    let temp1 = 1e1;
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
        let neigh_type = rng.gen_range(0..10);

        let new_state = if neigh_type < 8 {
            if state.len() <= 1 {
                continue;
            }

            let i = rng.gen_range(0..state.len());
            let mut j = rng.gen_range(0..state.len() - 1);

            if j >= i {
                j += 1;
            }

            if state.widths[i] <= 1 {
                continue;
            }

            let mut new_state = state.clone();
            let dw = rng.gen_range(1..state.widths[i]);

            new_state.widths[i] -= dw;
            new_state.widths[j] += dw;
            new_state
        } else if neigh_type == 8 {
            if state.len() >= input.n {
                continue;
            }

            let i = rng.gen_range(0..state.len());

            if state.widths[i] <= 1 {
                continue;
            }

            let mut new_state = state.clone();
            let dw = rng.gen_range(1..state.widths[i]);
            new_state.widths[i] -= dw;
            new_state.widths.push(dw);
            new_state
        } else {
            if state.len() <= 1 {
                continue;
            }

            let i = rng.gen_range(0..state.len());
            let mut j = rng.gen_range(0..state.len() - 1);

            if j >= i {
                j += 1;
            }

            let mut new_state = state.clone();
            new_state.widths[i] += new_state.widths[j];
            new_state.widths.remove(j);
            new_state
        };

        // スコア計算
        let new_score = new_state.calc_score(input);
        let score_diff = new_score - current_score;

        if score_diff <= 0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;
            state = new_state;

            if best_score.change_min(current_score) {
                best_state = state.clone();
                update_count += 1;
            }
        }

        valid_iter += 1;
    }

    eprintln!("===== annealing =====");
    eprintln!("init score : {}", init_score);
    eprintln!("score      : {}", best_score);
    eprintln!("all iter   : {}", all_iter);
    eprintln!("valid iter : {}", valid_iter);
    eprintln!("accepted   : {}", accepted_count);
    eprintln!("updated    : {}", update_count);
    eprintln!("");

    best_state
}
