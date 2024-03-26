use itertools::Itertools;
use rand::prelude::*;

use crate::{
    common::ChangeMinMax as _,
    problem::{Input, Rect},
};

pub fn devide(input: &Input, dividers: &[i32]) -> Vec<Vec<Rect>> {
    let env = Env::new(input, dividers);

    let separator_count = input.n - (dividers.len() - 1);
    let lines = (1..=separator_count)
        .map(|i| Separator::new(0, i as i32))
        .collect_vec();
    let state = State::new(vec![lines; input.days]);

    let duration = 2.9 - input.since.elapsed().as_secs_f64();
    let state = annealing(&env, state, duration);

    state.to_rects(&env)
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
}

impl State {
    fn new(mut lines: Vec<Vec<Separator>>) -> Self {
        for lines in lines.iter_mut() {
            glidesort::sort(lines);
        }

        Self { lines }
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

    fn calc_day_score(&mut self, env: &Env, day: usize) -> Result<i64, ()> {
        glidesort::sort(&mut self.lines[day]);

        let mut score = 0;
        score += self.calc_area_score(env, day)?;
        score += self.calc_line_score(env, day);
        score += self.calc_line_score(env, day + 1);

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

        glidesort::sort(&mut areas);

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

                assert!(rect.is_valid());

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
    let mut current_score = state.calc_score(&env).unwrap();
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

    let temp0 = 1e9;
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

        // スコア計算
        let old_score = state.calc_day_score(env, day).unwrap();

        std::mem::swap(&mut state.lines[day], &mut new_lines);

        let Ok(new_score) = state.calc_day_score(&env, day) else {
            std::mem::swap(&mut state.lines[day], &mut new_lines);
            continue;
        };
        let score_diff = new_score - old_score;

        if score_diff <= 0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score += score_diff;
            accepted_count += 1;

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
