use crate::{
    common::{ChangeMinMax as _, DIVISOR},
    problem::Input,
};
use rand::prelude::*;

pub fn generate_init_solution(input: &Input, duration: f64) -> (Vec<i32>, Vec<Vec<(usize, i32)>>) {
    let state = State::new(vec![Input::W], vec![false; input.n]);
    let state = annealing(input, state, duration);
    let score = state.calc_score(input, i64::MAX);
    eprintln!("first_fit - step1 score: {}", score);

    state.restore(input)
}

#[derive(Debug, Clone)]
struct State {
    widths: Vec<i32>,
    directions: Vec<bool>,
}

impl State {
    fn new(widths: Vec<i32>, directions: Vec<bool>) -> Self {
        Self { widths, directions }
    }

    fn len(&self) -> usize {
        self.widths.len()
    }

    fn calc_score(&self, input: &Input, threshold: i64) -> i64 {
        let mut widths = self.widths.clone();
        widths.sort_unstable();

        let mut y_lanes = vec![0; widths.len()];

        let mut score = 1;

        for (day, requests) in input.requests.iter().enumerate() {
            let line_mul = if day == 0 || day == input.days - 1 {
                1
            } else {
                2
            };

            y_lanes.fill(0);

            for (rect_j, &request) in requests.iter().enumerate().rev() {
                let mut next_i = !0;
                let mut next_y = i32::MAX;

                // first-fitをする
                if self.directions[rect_j] {
                    for (i, (&width, &y)) in widths.iter().zip(y_lanes.iter()).enumerate().rev() {
                        let dy = DIVISOR.div((request + width - 1) as u32, width as u32) as i32;
                        let y = y + dy;

                        if next_y.change_min(y) {
                            next_i = i;
                        }

                        if next_y <= Input::W {
                            break;
                        }
                    }
                } else {
                    for (i, (&width, &y)) in widths.iter().zip(y_lanes.iter()).enumerate() {
                        let dy = DIVISOR.div((request + width - 1) as u32, width as u32) as i32;
                        let y = y + dy;

                        if next_y.change_min(y) {
                            next_i = i;
                        }

                        if next_y <= Input::W {
                            break;
                        }
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

            // 1つ以上入れないとダメ
            score += y_lanes.iter().filter(|&&y| y == 0).count() as i64 * 1000;

            // 閾値以下にならないことが確定したら打ち切り
            if score > threshold {
                return 1 << 50;
            }
        }

        score
    }

    fn restore(&self, input: &Input) -> (Vec<i32>, Vec<Vec<(usize, i32)>>) {
        let dividers = self.restore_dividers();
        let separatators = self.restore_separators(input);
        (dividers, separatators)
    }

    fn restore_dividers(&self) -> Vec<i32> {
        let mut x = 0;
        let mut dividers = vec![0];

        let mut widths = self.widths.clone();
        widths.sort_unstable();

        for w in widths.iter() {
            x += *w;
            dividers.push(x);
        }

        dividers
    }

    fn restore_separators(&self, input: &Input) -> Vec<Vec<(usize, i32)>> {
        let mut all_separators = vec![];
        let mut widths = self.widths.clone();
        widths.sort();

        let mut y_lanes = vec![0; widths.len()];

        for requests in input.requests.iter() {
            let mut lane_rects = vec![vec![]; widths.len()];
            y_lanes.fill(0);

            for (rect_j, &request) in requests.iter().enumerate().rev() {
                let mut next_i = !0;
                let mut next_y = i32::MAX;

                // first-fitをする
                if self.directions[rect_j] {
                    for (i, (&width, &y)) in widths.iter().zip(y_lanes.iter()).enumerate().rev() {
                        let dy = DIVISOR.div((request + width - 1) as u32, width as u32) as i32;
                        let y = y + dy;

                        if next_y.change_min(y) {
                            next_i = i;
                        }

                        if next_y <= Input::W {
                            break;
                        }
                    }
                } else {
                    for (i, (&width, &y)) in widths.iter().zip(y_lanes.iter()).enumerate() {
                        let dy = DIVISOR.div((request + width - 1) as u32, width as u32) as i32;
                        let y = y + dy;

                        if next_y.change_min(y) {
                            next_i = i;
                        }

                        if next_y <= Input::W {
                            break;
                        }
                    }
                }

                lane_rects[next_i].push(next_y - y_lanes[next_i]);
                y_lanes[next_i] = next_y;
            }

            let mut separators = vec![];

            for i in 0..widths.len() {
                let mut y = 0;
                let mut local_separators = vec![];

                for &dy in lane_rects[i].iter() {
                    y += dy;
                    local_separators.push(y);
                }

                if let Some(&last) = local_separators.last() {
                    for y in local_separators.iter_mut() {
                        *y *= Input::W;
                        *y /= last;
                    }

                    local_separators.pop();

                    for &y in local_separators.iter() {
                        separators.push((i, y));
                    }
                }
            }

            let separator_count = input.n - self.widths.len();

            if separators.len() > separator_count {
                separators.truncate(separator_count);
            }

            all_separators.push(separators);
        }

        all_separators
    }
}

fn annealing(input: &Input, initial_solution: State, duration: f64) -> State {
    let mut state = initial_solution;
    let mut best_state = state.clone();
    let mut current_score = state.calc_score(input, i64::MAX);
    let mut best_score = current_score;

    let mut all_iter = 0;
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e7;
    let temp1 = 1e0;
    let mut temp = temp0;

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let neigh_type = rng.gen_range(0..15);

        let mut new_state = if neigh_type < 8 {
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
            if state.len() >= input.n - 1 {
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
        } else if neigh_type == 9 {
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
        } else {
            let i = rng.gen_range(0..state.directions.len());
            let mut new_state = state.clone();
            new_state.directions[i] ^= true;
            new_state
        };

        new_state.widths.sort_unstable();

        // スコア計算
        // 先に閾値を求めることで評価を高速化する
        let score_threshold =
            current_score - (temp * rng.gen_range(0.0f64..1.0).ln()).round() as i64;

        let new_score = new_state.calc_score(input, score_threshold);

        if new_score <= score_threshold {
            // 解の更新
            current_score = new_score;
            state = new_state;

            if best_score.change_min(current_score) {
                best_state = state.clone();
            }
        }
    }

    eprintln!("all iter: {}", all_iter);

    best_state
}
