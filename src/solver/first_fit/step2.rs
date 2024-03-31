use itertools::Itertools;
use rand::prelude::*;

use crate::{
    common::ChangeMinMax as _,
    problem::{Input, Rect},
};

const INF: i64 = 1 << 50;

pub fn devide(
    input: &Input,
    dividers: &[i32],
    separators: Vec<Vec<(usize, i32)>>,
) -> (Vec<Vec<Rect>>, i64) {
    let env = Env::new(input, dividers);

    let separators = separators
        .into_iter()
        .map(|seps| {
            seps.into_iter()
                .map(|(i, y)| Separator::new(i, y))
                .collect_vec()
        })
        .collect_vec();
    eprintln!("{:?}", separators);
    let mut state = State::new(separators, &env);
    eprintln!("init score: {:?}", state.calc_score(&env).unwrap_or(INF));

    let duration = 1.35;
    let mut state = annealing(&env, state, duration);
    let score = state.calc_score(&env).unwrap_or(INF);

    (state.to_rects(&env), score)
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    dividers: &'a [i32],
    widths: Vec<i32>,
}

impl<'a> Env<'a> {
    fn new(input: &'a Input, dividers: &'a [i32]) -> Self {
        let widths = dividers
            .iter()
            .tuple_windows()
            .map(|(&a, &b)| b - a)
            .collect_vec();

        Self {
            input,
            dividers,
            widths,
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    lines: Vec<Vec<Separator>>,
    area_scores: Vec<i64>,
    line_scores: Vec<i64>,
}

impl State {
    fn new(mut lines: Vec<Vec<Separator>>, env: &Env) -> Self {
        for lines in lines.iter_mut() {
            glidesort::sort(lines);
        }

        let mut state = Self {
            lines,
            area_scores: vec![],
            line_scores: vec![],
        };

        for day in 0..env.input.days {
            let s = state.calc_area_score(env, day).unwrap_or(INF);
            state.area_scores.push(s);
        }

        for day in 0..env.input.days + 1 {
            let s = state.calc_line_score(env, day);
            state.line_scores.push(s);
        }

        state
    }

    fn calc_score(&mut self, env: &Env) -> Result<i64, ()> {
        let mut score = 1;

        for lines in self.lines.iter_mut() {
            glidesort::sort(lines);
        }

        for day in 0..env.input.days {
            score += self.calc_area_score(env, day)?;
            score += self.calc_line_score(env, day);
        }

        Ok(score)
    }

    fn calc_day_score(&self, env: &Env, day: usize, threshold: i64) -> Result<i64, ()> {
        let mut score = 0;
        score += self.calc_line_score(env, day);

        if score > threshold {
            return Err(());
        }

        score += self.calc_line_score(env, day + 1);

        if score > threshold {
            return Err(());
        }

        score += self.calc_area_score(env, day)?;

        Ok(score)
    }

    fn calc_area_score(&self, env: &Env, day: usize) -> Result<i64, ()> {
        let mut areas = Vec::with_capacity(env.input.n);
        let lines = &self.lines[day];

        let mut pointer = 0;

        for (i, &width) in env.widths.iter().enumerate() {
            let mut y = 0;

            while pointer < lines.len() && lines[pointer].index == i {
                let area = (lines[pointer].y - y) * width;
                y = lines[pointer].y;

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

        for (&req, &area) in env.input.requests[day].iter().zip(areas.iter()) {
            let diff = (req - area) as i64;
            score += diff.max(0) * 100;
        }

        Ok(score)
    }

    fn calc_line_score(&self, env: &Env, day: usize) -> i64 {
        let Some(lines0) = self.lines.get(day.wrapping_sub(1)) else {
            return 0;
        };
        let Some(lines1) = self.lines.get(day) else {
            return 0;
        };

        let mut score = 0;
        let mut ptr0 = 0;
        let mut ptr1 = 0;

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

    fn to_rects(&self, env: &Env) -> Vec<Vec<Rect>> {
        let mut all_rects = vec![];

        for day in 0..env.input.days {
            let mut rects = vec![];
            let mut pointer = 0;
            let lines = &self.lines[day];

            for (i, (&x0, &x1)) in env.dividers.iter().tuple_windows().enumerate() {
                let mut y = 0;

                while pointer < lines.len() && lines[pointer].index == i {
                    let rect = Rect::new(x0, y, x1, lines[pointer].y);
                    y = lines[pointer].y;

                    assert!(rect.is_valid());

                    rects.push(rect);
                    pointer += 1;
                }

                let rect = Rect::new(x0, y, x1, Input::W);

                rects.push(rect);
            }

            glidesort::sort_by_key(&mut rects, |r| r.area());

            all_rects.push(rects);
        }

        all_rects
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
    let mut current_score = state.calc_score(&env).unwrap_or(INF);
    let init_score = current_score;
    let mut best_solution = state.clone();
    let mut best_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::from_entropy();

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e2;
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
        let neigh_type = rng.gen_range(0..5);
        let day = rng.gen_range(0..env.input.days);

        let mut new_lines = if neigh_type == 0 {
            let index = rng.gen_range(0..state.lines[day].len());
            let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
            let dy = sign * 10f64.powf(rng.gen_range(0.0..3.0)).round() as i32;
            let mut new_lines = state.lines[day].clone();
            new_lines[index].y += dy;
            new_lines
        } else if neigh_type == 1 {
            let index = rng.gen_range(0..state.lines[day].len());
            let new_index = rng.gen_range(0..env.widths.len());
            let new_y = rng.gen_range(1..Input::W);
            let mut new_lines = state.lines[day].clone();
            new_lines[index] = Separator::new(new_index, new_y);
            new_lines
        } else if neigh_type == 2 {
            let day_diff = if rng.gen_bool(0.5) { 1 } else { -1 };

            let Some(target_lines) = state.lines.get(day.wrapping_add_signed(day_diff)) else {
                continue;
            };

            let i0 = rng.gen_range(0..target_lines.len());
            let i1 = rng.gen_range(0..state.lines[day].len());
            let mut new_lines = state.lines[day].clone();
            new_lines[i1] = target_lines[i0];
            new_lines
        } else if neigh_type == 3 {
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

            let mut new_lines = state.lines[day].clone();

            for sep in new_lines.iter_mut() {
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

            new_lines
        } else {
            let i = rng.gen_range(0..env.widths.len());

            let mut targets = vec![];

            for (j, l) in state.lines[day].iter().enumerate() {
                if l.index == i {
                    targets.push(j);
                }
            }

            if targets.len() == 0 {
                continue;
            }

            glidesort::sort_by_key(&mut targets, |j| state.lines[day][*j].y);

            let mut y = 0;
            let mut dy = vec![];

            for &j in targets.iter() {
                dy.push(state.lines[day][j].y - y);
                y = state.lines[day][j].y;
            }

            dy.push(Input::W - y);

            dy.shuffle(&mut rng);

            let mut y = 0;

            let mut new_lines = state.lines[day].clone();

            for (&j, &dy) in targets.iter().zip(dy.iter()) {
                y += dy;
                new_lines[j].y = y;
            }

            new_lines
        };
        new_lines.sort_unstable();

        // スコア計算
        // 先に閾値を求めることで評価を高速化する
        let old_score =
            state.area_scores[day] + state.line_scores[day] + state.line_scores[day + 1];
        let score_threshold = old_score as f64 - temp * rng.gen_range(0.0f64..1.0).ln();

        std::mem::swap(&mut state.lines[day], &mut new_lines);

        let prev_line_score = state.calc_line_score(env, day);

        if prev_line_score as f64 > score_threshold {
            std::mem::swap(&mut state.lines[day], &mut new_lines);
            continue;
        }

        let next_line_score = state.calc_line_score(env, day + 1);

        if (prev_line_score + next_line_score) as f64 > score_threshold {
            std::mem::swap(&mut state.lines[day], &mut new_lines);
            continue;
        }

        let Ok(area_score) = state.calc_area_score(&env, day) else {
            std::mem::swap(&mut state.lines[day], &mut new_lines);
            continue;
        };

        let new_score = area_score + prev_line_score + next_line_score;
        let score_diff = new_score - old_score;

        if new_score as f64 <= score_threshold {
            // 解の更新
            current_score += score_diff;
            accepted_count += 1;
            state.area_scores[day] = area_score;
            state.line_scores[day] = prev_line_score;
            state.line_scores[day + 1] = next_line_score;

            if best_score.change_min(current_score) {
                best_solution = state.clone();
                update_count += 1;
            }
        } else {
            std::mem::swap(&mut state.lines[day], &mut new_lines);
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

    best_solution
}
