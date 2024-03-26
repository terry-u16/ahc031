use itertools::Itertools;
use rand::prelude::*;

use crate::{
    common::ChangeMinMax as _,
    problem::{Input, Rect},
};

pub fn devide(input: &Input, dividers: &[i32]) -> Vec<Vec<Rect>> {
    let mut rects = vec![];
    let mut prev_lines = None;
    let mut prev_reqs = None;
    let each_duration = (2.9 - input.since.elapsed().as_secs_f64()) / input.days as f64;
    let mut total_score = 1;

    for day in 0..input.days {
        let env = Env::new(&input, dividers, day, prev_lines.clone());
        let reqs = prev_reqs.take().unwrap_or_else(|| {
            let mut reqs = vec![vec![]; env.widths.len()];

            for i in 0..input.n {
                reqs[i % env.widths.len()].push(i);
            }

            reqs
        });

        let state = State::new(reqs);
        let state = annealing(&env, state, each_duration);
        let score = state.calc_score(&env);
        total_score += score;
        prev_lines = Some(state.restore_lines(&env));
        rects.push(state.to_rect(&env));
        prev_reqs = Some(state.requests);
    }

    eprintln!("total score : {}", total_score);

    rects
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    dividers: &'a [i32],
    widths: Vec<i32>,
    day: usize,
    prev_y: Option<Vec<Vec<i32>>>,
}

impl<'a> Env<'a> {
    fn new(
        input: &'a Input,
        dividers: &'a [i32],
        day: usize,
        prev_y: Option<Vec<Vec<i32>>>,
    ) -> Self {
        let widths = dividers
            .iter()
            .tuple_windows()
            .map(|(&a, &b)| b - a)
            .collect_vec();

        Self {
            input,
            dividers,
            widths,
            day,
            prev_y,
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    requests: Vec<Vec<usize>>,
}

impl State {
    fn new(requests: Vec<Vec<usize>>) -> Self {
        Self { requests }
    }

    fn calc_score(&self, env: &Env) -> i32 {
        let mut score = 0;

        for lane in 0..env.widths.len() {
            score += self.calc_score_lane(env, lane);
        }

        eprintln!();

        score
    }

    fn calc_score_lane(&self, env: &Env, lane: usize) -> i32 {
        const INF: i32 = i32::MAX / 2;
        let Some(mut prev_lines) = env.prev_y.as_ref().map(|v| v[lane].clone()) else {
            return self.calc_score_lane_day0(env, lane);
        };

        let width = env.widths[lane];
        prev_lines.insert(0, 0);
        let requests = &self.requests[lane];
        let mut dp = vec![vec![INF; prev_lines.len()]; 1 << requests.len()];
        let mut y = vec![vec![0; prev_lines.len()]; 1 << requests.len()];
        dp[0][0] = 0;

        for flag in 0usize..1 << requests.len() {
            let placed_cnt = flag.count_ones() as usize;

            for i in 0..prev_lines.len() {
                let yy = y[flag][i];
                let current_cost = dp[flag][i];

                for j in 0..requests.len() {
                    if ((flag >> j) & 1) > 0 {
                        continue;
                    }

                    let next_flag = flag | (1 << j);
                    let request = requests[j];
                    let area = env.input.requests[env.day][request];
                    let next_y = yy + (area + width - 1) / width;
                    let area_cost = (next_y - yy.max(Input::W)).max(0) * width * Input::AREA_MUL;

                    // 何本目のラインまで跨ぎ終わったかを調べる
                    // ピッタリライン上のものは跨ぎ終わっていない判定
                    let mut next_i = i;

                    for i in i + 1..prev_lines.len() {
                        if prev_lines[i] >= next_y {
                            break;
                        }

                        next_i = i;
                    }

                    // 跨いだラインの数だけ削除コストがかかる
                    let line_cost = (next_i - i) as i32 * width;
                    let end_cost = if placed_cnt == requests.len() - 1 {
                        0
                    } else {
                        width
                    };

                    // 長方形が収まるピッタリの位置にラインを引く遷移
                    let next_cost = current_cost + area_cost + line_cost + end_cost;
                    if dp[next_flag][next_i].change_min(next_cost) {
                        y[next_flag][next_i] = next_y;
                    }

                    // 1個先のラインに合わせてラインを引く遷移
                    if let Some(&next_y) = prev_lines.get(next_i + 1) {
                        if placed_cnt == requests.len() - 1 {
                            continue;
                        }

                        let next_cost = current_cost + area_cost + line_cost;

                        if dp[next_flag][next_i + 1].change_min(next_cost) {
                            y[next_flag][next_i + 1] = next_y;
                        }
                    }
                }
            }
        }

        let mut best_cost = INF;
        let flag = (1 << requests.len()) - 1;

        for i in 0..prev_lines.len() {
            // ラインを使っていない場合は削除コストがかかることに注意
            best_cost.change_min(dp[flag][i] + (prev_lines.len() - i - 1) as i32 * width);
        }

        best_cost
    }

    fn calc_score_lane_day0(&self, env: &Env, lane: usize) -> i32 {
        // 初日の場合はラインのコストがラインの引き方に依らない
        let mut y = 0;
        let width = env.widths[lane];

        for &lane in self.requests[lane].iter() {
            let request = env.input.requests[env.day][lane];
            y += (request + width - 1) / width;
        }

        let area_cost = (y - Input::W).max(0) * Input::AREA_MUL;

        area_cost
    }

    fn restore_lines(&self, env: &Env) -> Vec<Vec<i32>> {
        let mut lines = vec![];

        for lane in 0..env.widths.len() {
            lines.push(self.restore_lines_lane(env, lane));
        }

        lines
    }

    fn restore_lines_lane(&self, env: &Env, lane: usize) -> Vec<i32> {
        const INF: i32 = i32::MAX / 2;
        let Some(mut prev_lines) = env.prev_y.as_ref().map(|v| v[lane].clone()) else {
            return self.restore_lines_lane_day0(env, lane);
        };

        let width = env.widths[lane];
        prev_lines.insert(0, 0);
        let requests = &self.requests[lane];
        let mut dp = vec![vec![INF; prev_lines.len()]; 1 << requests.len()];
        let mut y = vec![vec![0; prev_lines.len()]; 1 << requests.len()];
        let mut from = vec![vec![(!0, !0); prev_lines.len()]; 1 << requests.len()];
        dp[0][0] = 0;

        for flag in 0usize..1 << requests.len() {
            let placed_cnt = flag.count_ones() as usize;

            for i in 0..prev_lines.len() {
                let yy = y[flag][i];
                let current_cost = dp[flag][i];

                for j in 0..requests.len() {
                    if ((flag >> j) & 1) > 0 {
                        continue;
                    }

                    let next_flag = flag | (1 << j);
                    let request = requests[j];
                    let area = env.input.requests[env.day][request];
                    let next_y = yy + (area + width - 1) / width;
                    let area_cost = (next_y - yy.max(Input::W)).max(0) * width * Input::AREA_MUL;

                    // 何本目のラインまで跨ぎ終わったかを調べる
                    // ピッタリライン上のものは跨ぎ終わっていない判定
                    let mut next_i = i;

                    for i in i + 1..prev_lines.len() {
                        if prev_lines[i] >= next_y {
                            break;
                        }

                        next_i = i;
                    }

                    // 跨いだラインの数だけ削除コストがかかる
                    let line_cost = (next_i - i) as i32 * width;
                    let end_cost = if placed_cnt == requests.len() - 1 {
                        0
                    } else {
                        width
                    };

                    // 長方形が収まるピッタリの位置にラインを引く遷移
                    let next_cost = current_cost + area_cost + line_cost + end_cost;
                    if dp[next_flag][next_i].change_min(next_cost) {
                        y[next_flag][next_i] = next_y;
                        from[next_flag][next_i] = (flag, i);
                    }

                    // 1個先のラインに合わせてラインを引く遷移
                    if let Some(&next_y) = prev_lines.get(next_i + 1) {
                        if placed_cnt == requests.len() - 1 {
                            continue;
                        }

                        let next_cost = current_cost + area_cost + line_cost;

                        if dp[next_flag][next_i + 1].change_min(next_cost) {
                            y[next_flag][next_i + 1] = next_y;
                            from[next_flag][next_i + 1] = (flag, i);
                        }
                    }
                }
            }
        }

        let mut best_cost = INF;
        let flag = (1 << requests.len()) - 1;
        let mut best_i = 0;

        for i in 0..prev_lines.len() {
            // ラインを使っていない場合は削除コストがかかることに注意
            let cost = dp[flag][i] + (prev_lines.len() - i - 1) as i32 * width;

            if best_cost.change_min(cost) {
                best_i = i;
            }
        }

        let mut lines = vec![];
        let mut flag = (1 << requests.len()) - 1;
        let mut i = best_i;

        loop {
            (flag, i) = from[flag][i];

            if flag == 0 {
                break;
            }

            lines.push(y[flag][i]);
        }

        lines.reverse();

        lines
    }

    fn restore_lines_lane_day0(&self, env: &Env, lane: usize) -> Vec<i32> {
        // 初日の場合はラインのコストがラインの引き方に依らない
        let mut y = 0;
        let width = env.widths[lane];
        let mut lines = vec![];
        let len = self.requests[lane].len();

        for &lane in self.requests[lane][..len - 1].iter() {
            let request = env.input.requests[env.day][lane];
            y += (request + width - 1) / width;
            lines.push(y);
        }

        lines
    }

    fn to_rect(&self, env: &Env) -> Vec<Rect> {
        let mut rects = vec![];

        for lane in 0..env.widths.len() {
            let mut lines = self.restore_lines_lane(env, lane);
            lines.push(0);
            lines.push(Input::W);
            glidesort::sort(&mut lines);

            for (&y0, &y1) in lines.iter().tuple_windows() {
                rects.push(Rect::new(
                    env.dividers[lane],
                    y0,
                    env.dividers[lane + 1],
                    y1,
                ));
            }
        }

        glidesort::sort_by_key(&mut rects, |r| r.area());

        rects
    }
}

fn annealing(env: &Env, mut state: State, duration: f64) -> State {
    let mut best_state = state.clone();
    let mut current_score = state.calc_score(env);
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
    let temp1 = 1e2;
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
        let neigh_type = rng.gen_range(0..2);
        let i0 = rng.gen_range(0..state.requests.len());
        let mut i1 = rng.gen_range(0..state.requests.len() - 1);
        if i1 >= i0 {
            i1 += 1;
        }

        let new_state = if neigh_type == 0 {
            let j0 = rng.gen_range(0..state.requests[i0].len());
            let j1 = rng.gen_range(0..state.requests[i1].len());
            let mut new_state = state.clone();
            let temp = new_state.requests[i0][j0];
            new_state.requests[i0][j0] = new_state.requests[i1][j1];
            new_state.requests[i1][j1] = temp;
            new_state
        } else {
            if state.requests[i0].len() == 1 {
                continue;
            }

            let j0 = rng.gen_range(0..state.requests[i0].len());
            let mut new_state = state.clone();
            let temp = new_state.requests[i0].swap_remove(j0);
            new_state.requests[i1].push(temp);
            new_state
        };

        // スコア計算
        let score_diff = new_state.calc_score_lane(env, i0) + new_state.calc_score_lane(env, i1)
            - state.calc_score_lane(env, i0)
            - state.calc_score_lane(env, i1);

        if score_diff <= 0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score += score_diff;
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
