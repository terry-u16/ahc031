use std::time::Instant;

use itertools::Itertools;
use rand::prelude::*;

use crate::{
    common::ChangeMinMax as _,
    problem::{Input, Rect},
};

pub fn divide(input: &Input, dividers: &[Vec<i32>], duration: f64) -> (Vec<Vec<Rect>>, i64) {
    let since = Instant::now();
    let trial_count = (3000 / (input.days * input.n)).max(3);
    let max_beam_width = trial_count / 2;

    let separator_count = input.n - (dividers[0].len() - 1);
    let lines = (1..=separator_count)
        .map(|i| Separator::new(0, i as i32))
        .collect_vec();
    let init_state = State::new(lines);

    let mut beam = vec![vec![]; input.days + 1];

    for (i, dividers) in dividers.iter().enumerate() {
        let env = Env::new(&input, dividers, 0, None);
        beam[0].push(BeamState::new(&env, None, 1, i, !0));
    }

    for day in 0..input.days {
        let each_duration =
            ((duration - since.elapsed().as_secs_f64()) / (input.days - day) as f64).max(0.0);

        for i in 0..trial_count {
            let beam_index = i % beam[day].len();
            let beam_state = &beam[day][beam_index];
            let prev_state = beam_state.state.clone();
            let prev_score = beam_state.score;
            let divider_index = beam_state.divider_index;
            let state = prev_state.clone().unwrap_or_else(|| init_state.clone());
            let env = Env::new(&input, &dividers[beam_state.divider_index], day, prev_state);

            let duration = each_duration / trial_count as f64;
            let state = annealing(&env, state.clone(), duration);

            beam[day + 1].push(BeamState::new(
                &env,
                Some(state),
                prev_score,
                divider_index,
                beam_index,
            ));
        }

        beam[day + 1].sort_unstable_by_key(|s| s.score);

        if beam[day + 1].len() > max_beam_width {
            beam[day + 1].truncate(max_beam_width);
        }
    }

    let best_state = &beam[input.days][0];
    let mut rects = vec![];
    let mut day = input.days;
    let mut index = 0;

    while day > 0 {
        let state = &beam[day][index];
        let prev_day = day - 1;
        let prev_index = state.prev_state_index;
        let prev_state = beam[prev_day][prev_index].state.clone();
        let env = Env::new(&input, &dividers[state.divider_index], day - 1, prev_state);
        rects.push(state.state.as_ref().unwrap().to_rects(&env));
        day = prev_day;
        index = prev_index;
    }

    rects.reverse();

    (rects, best_state.score)
}

#[derive(Debug, Clone)]
struct BeamState {
    state: Option<State>,
    divider_index: usize,
    score: i64,
    prev_state_index: usize,
}

impl BeamState {
    fn new(
        env: &Env,
        mut state: Option<State>,
        prev_score: i64,
        divider_index: usize,
        prev_state_index: usize,
    ) -> Self {
        let score = state
            .as_mut()
            .map(|s| s.calc_score(env).unwrap())
            .unwrap_or(0)
            + prev_score;
        Self {
            state,
            score,
            divider_index,
            prev_state_index,
        }
    }
}

impl PartialEq for BeamState {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for BeamState {}

impl PartialOrd for BeamState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for BeamState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    dividers: &'a [i32],
    widths: Vec<i32>,
    day: usize,
    prev_state: Option<State>,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input, dividers: &'a [i32], day: usize, prev_state: Option<State>) -> Self {
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
            prev_state,
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    lines: Vec<Separator>,
}

impl State {
    fn new(mut lines: Vec<Separator>) -> Self {
        lines.sort_unstable();
        Self { lines }
    }

    fn calc_score(&mut self, env: &Env) -> Result<i64, ()> {
        let mut score = self.calc_area_score(env)?;

        if let Some(prev_state) = &env.prev_state {
            score += self.calc_line_score(env, prev_state);
        }

        Ok(score)
    }

    fn calc_area_score(&self, env: &Env<'_>) -> Result<i64, ()> {
        let mut areas = Vec::with_capacity(env.input.n);
        let mut pointer = 0;

        for (i, &width) in env.widths.iter().enumerate() {
            let mut y = 0;

            while pointer < self.lines.len() && self.lines[pointer].index == i {
                let area = (self.lines[pointer].y - y) * width;
                y = self.lines[pointer].y;

                if area <= 0 {
                    return Err(());
                }

                areas.push(area);
                pointer += 1;
            }

            let area = (Input::W - y) * width;

            if area <= 0 {
                return Err(());
            }

            areas.push(area);
        }

        areas.sort_unstable();

        let mut score = 0;

        for (&req, &area) in env.input.requests[env.day].iter().zip(areas.iter()) {
            let diff = req - area;
            score += diff.max(0);
        }

        Ok(score as i64 * 100)
    }

    fn calc_line_score(&self, env: &Env, prev_state: &State) -> i64 {
        let mut score = 0;
        let mut ptr0 = 0;
        let mut ptr1 = 0;

        let lines0 = &prev_state.lines;
        let lines1 = &self.lines;
        let inf = Separator::new(usize::MAX, i32::MAX);

        while ptr0 < lines0.len() || ptr1 < lines1.len() {
            let line0 = lines0.get(ptr0).copied().unwrap_or(inf);
            let line1 = lines1.get(ptr1).copied().unwrap_or(inf);

            if line0 == line1 {
                ptr0 += 1;
                ptr1 += 1;
            } else if line0 < line1 {
                score += env.widths[line0.index];
                ptr0 += 1;
            } else {
                score += env.widths[line1.index];
                ptr1 += 1;
            }
        }

        score as i64
    }

    fn to_rects(&self, env: &Env) -> Vec<Rect> {
        let mut rects = vec![];
        let mut pointer = 0;

        for (i, (&x0, &x1)) in env.dividers.iter().tuple_windows().enumerate() {
            let mut y = 0;

            while pointer < self.lines.len() && self.lines[pointer].index == i {
                let rect = Rect::new(x0, y, x1, self.lines[pointer].y);
                y = self.lines[pointer].y;

                assert!(rect.is_valid());

                rects.push(rect);
                pointer += 1;
            }

            let rect = Rect::new(x0, y, x1, Input::W);

            assert!(rect.is_valid());

            rects.push(rect);
        }

        glidesort::sort_by_key(&mut rects, |r| r.area());

        rects
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Separator {
    index: usize,
    y: i32,
}

impl Separator {
    fn new(index: usize, y: i32) -> Self {
        Self { index, y }
    }
}

fn annealing(env: &Env, mut state: State, duration: f64) -> State {
    if duration <= 0.0 {
        return state;
    }

    let mut current_score = state.calc_score(&env).unwrap();
    let mut best_solution = state.clone();
    let mut best_score = current_score;

    let mut all_iter = 0;
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e6;
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
        let neigh_type = rng.gen_range(0..10);

        let mut new_state = if neigh_type < 5 {
            let mut index = rng.gen_range(0..state.lines.len());
            let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
            let dy = sign * 10f64.powf(rng.gen_range(0.0..3.0)).round() as i32;
            let mut new_state = state.clone();
            new_state.lines[index].y += dy;

            while index > 0 && new_state.lines[index] < new_state.lines[index - 1] {
                new_state.lines.swap(index, index - 1);
                index -= 1;
            }

            while index + 1 < new_state.lines.len()
                && new_state.lines[index] > new_state.lines[index + 1]
            {
                new_state.lines.swap(index, index + 1);
                index += 1;
            }

            new_state
        } else if neigh_type < 7 {
            let mut index = rng.gen_range(0..state.lines.len());
            let new_index = rng.gen_range(0..env.widths.len());
            let new_y = rng.gen_range(1..Input::W);
            let mut new_state = state.clone();
            new_state.lines[index] = Separator::new(new_index, new_y);

            while index > 0 && new_state.lines[index] < new_state.lines[index - 1] {
                new_state.lines.swap(index, index - 1);
                index -= 1;
            }

            while index + 1 < new_state.lines.len()
                && new_state.lines[index] > new_state.lines[index + 1]
            {
                new_state.lines.swap(index, index + 1);
                index += 1;
            }

            new_state
        } else if neigh_type == 7 {
            let Some(prev_state) = &env.prev_state else {
                continue;
            };

            let i0 = rng.gen_range(0..prev_state.lines.len());
            let i1 = rng.gen_range(0..state.lines.len());
            let mut new_state = state.clone();
            new_state.lines[i1] = prev_state.lines[i0];

            let mut index = i1;

            while index > 0 && new_state.lines[index] < new_state.lines[index - 1] {
                new_state.lines.swap(index, index - 1);
                index -= 1;
            }

            while index + 1 < new_state.lines.len()
                && new_state.lines[index] > new_state.lines[index + 1]
            {
                new_state.lines.swap(index, index + 1);
                index += 1;
            }

            new_state
        } else if neigh_type == 8 {
            let index0 = rng.gen_range(0..env.widths.len());
            let diff = rng.gen_range(1..=3);

            let index1 = if rng.gen_bool(0.5) {
                index0 + diff
            } else {
                index0.wrapping_sub(diff)
            };

            if index1 >= env.widths.len() {
                continue;
            }

            let mut new_state = state.clone();

            for sep in new_state.lines.iter_mut() {
                let index = sep.index;

                let new_sep = if index == index0 {
                    Separator::new(index1, sep.y)
                } else if index == index1 {
                    Separator::new(index0, sep.y)
                } else {
                    Separator::new(index, sep.y)
                };

                *sep = new_sep;
            }
            new_state.lines.sort_unstable();

            new_state
        } else {
            let i = rng.gen_range(0..env.widths.len());

            let mut targets = vec![];

            for (j, l) in state.lines.iter().enumerate() {
                if l.index == i {
                    targets.push(j);
                }
            }

            if targets.len() == 0 {
                continue;
            }

            glidesort::sort_by_key(&mut targets, |j| state.lines[*j].y);

            let mut y = 0;
            let mut dy = vec![];

            for &j in targets.iter() {
                dy.push(state.lines[j].y - y);
                y = state.lines[j].y;
            }

            dy.push(Input::W - y);

            dy.shuffle(&mut rng);

            let mut y = 0;

            let mut new_state = state.clone();

            for (&j, &dy) in targets.iter().zip(dy.iter()) {
                y += dy;
                new_state.lines[j].y = y;
            }

            new_state.lines.sort_unstable();

            new_state
        };

        // スコア計算
        let Ok(new_score) = new_state.calc_score(&env) else {
            continue;
        };
        let score_diff = new_score - current_score;

        if score_diff <= 0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            state = new_state;

            if best_score.change_min(current_score) {
                best_solution = state.clone();
            }
        }
    }

    best_solution
}
