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

static mut AREA_BUF: Vec<i32> = vec![];

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

    fn calc_area_score(&self, env: &Env, day: usize) -> Result<i64, ()> {
        let areas = unsafe {
            AREA_BUF.clear();
            &mut AREA_BUF
        };
        let lines = &self.lines[day];

        let mut pointer = 0;

        for (i, &width) in env.widths.iter().enumerate() {
            let mut y = 0;

            while pointer < lines.len() {
                let line = &lines[pointer];

                if line.index() != i {
                    break;
                }

                let area = (line.y - y) * width;
                y = line.y;

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
            score += diff.max(0);
        }

        Ok(score * 100)
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
        let mut line0 = lines0[ptr0];
        let mut line1 = lines1[ptr1];
        let n = lines0.len();

        while ptr0 < n || ptr1 < n {
            if line0 == line1 {
                ptr0 += 1;
                ptr1 += 1;
                line0 = if ptr0 < lines0.len() {
                    lines0[ptr0]
                } else {
                    inf
                };
                line1 = if ptr1 < lines1.len() {
                    lines1[ptr1]
                } else {
                    inf
                };
            } else if line0 < line1 {
                score += env.widths[line0.index()];
                ptr0 += 1;
                line0 = if ptr0 < lines0.len() {
                    lines0[ptr0]
                } else {
                    inf
                };
            } else {
                score += env.widths[line1.index()];
                ptr1 += 1;
                line1 = if ptr1 < lines1.len() {
                    lines1[ptr1]
                } else {
                    inf
                };
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

                while pointer < lines.len() && lines[pointer].index() == i {
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

#[derive(Debug, Clone, Copy, Hash)]
struct Separator {
    index: u32,
    y: i32,
    v: u64,
}

impl Separator {
    fn new(index: usize, y: i32) -> Self {
        let v = (index as u64) << 32 | (y + (1 << 16)) as u64;
        Self {
            index: index as u32,
            y,
            v,
        }
    }

    fn index(&self) -> usize {
        self.index as usize
    }
}

impl PartialEq for Separator {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Eq for Separator {}

impl PartialOrd for Separator {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.v.partial_cmp(&other.v)
    }
}

impl Ord for Separator {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.v.cmp(&other.v)
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
    let mut new_line_buffer = state.lines[0].clone();

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
        let neigh_type = rng.gen_range(0..4);
        let day = rng.gen_range(0..env.input.days);
        let index = rng.gen_range(0..state.lines[day].len());

        let mut new_lines = if neigh_type == 0 {
            let new_y = loop {
                let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
                let dy = sign * 10f64.powf(rng.gen_range(0.0..3.0)).round() as i32;
                let new_y = state.lines[day][index].y + dy;

                if 0 < new_y && new_y < Input::W {
                    break new_y;
                }
            };

            let new_lines = &mut new_line_buffer;
            new_lines.copy_from_slice(&state.lines[day]);
            new_lines[index] = Separator::new(state.lines[day][index].index(), new_y);
            new_lines
        } else if neigh_type == 1 {
            let new_index = rng.gen_range(0..env.widths.len());
            let new_y = rng.gen_range(1..Input::W);
            let new_lines = &mut new_line_buffer;
            new_lines.copy_from_slice(&state.lines[day]);
            new_lines[index] = Separator::new(new_index, new_y);
            new_lines
        } else {
            let day_diff = if rng.gen_bool(0.5) { 1 } else { -1 };

            let Some(target_lines) = state.lines.get(day.wrapping_add_signed(day_diff)) else {
                continue;
            };

            let i0 = rng.gen_range(0..target_lines.len());
            let new_lines = &mut new_line_buffer;
            new_lines.copy_from_slice(&state.lines[day]);
            new_lines[index] = target_lines[i0];
            new_lines
        };

        let line = new_lines.remove(index);
        let index = new_lines.binary_search(&line).unwrap_or_else(|i| i);
        new_lines.insert(index, line);

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
