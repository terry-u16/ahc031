use itertools::Itertools;
use rand::{Rng as _, SeedableRng};

use crate::{
    common::ChangeMinMax as _,
    problem::{Input, Rect},
};

pub fn devide(input: &Input, dividers: &[i32]) -> Vec<Vec<Rect>> {
    let mut rects = vec![];
    let mut prev_lines = None;
    let each_duration = (2.9 - input.since.elapsed().as_secs_f64()) / input.days as f64;

    for day in 0..input.days {
        let env = Env::new(&input, dividers, day, prev_lines.clone());
        let prev_l = prev_lines.unwrap_or_else(|| {
            let separator_count = input.n - (dividers.len() - 1);
            let lines = (1..=separator_count)
                .map(|i| Separator::new(0, i as i32))
                .collect_vec();
            lines
        });

        let day_len = if day < input.days - 1 { 2 } else { 1 };
        let v = vec![prev_l; day_len];
        let mut state = State::new(v);

        const TRIAL_COUNT: usize = 5;
        let mut best_score = state.calc_score(&env).unwrap();
        let mut best_state = state.clone();

        for _ in 0..TRIAL_COUNT {
            let duration = each_duration / TRIAL_COUNT as f64;
            let mut state = annealing(&env, state.clone(), duration);
            let score = state.calc_score(&env).unwrap();

            if best_score.change_min(score) {
                best_state = state;
            }
        }

        rects.push(best_state.to_rects(&env));
        prev_lines = Some(best_state.lines[0].clone());
    }

    rects
}

#[derive(Debug, Clone)]
struct Env<'a> {
    input: &'a Input,
    dividers: &'a [i32],
    widths: Vec<i32>,
    day: usize,
    prev_lines: Option<Vec<Separator>>,
}

impl<'a> Env<'a> {
    fn new(
        input: &'a Input,
        dividers: &'a [i32],
        day: usize,
        prev_lines: Option<Vec<Separator>>,
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
            prev_lines,
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
        for lines in self.lines.iter_mut() {
            glidesort::sort(lines);
        }

        let mut mul = 10;

        let mut prev_lines = env.prev_lines.as_ref();

        let mut day = env.day;
        let mut score = 0;

        for lines in self.lines.iter() {
            score += Self::calc_area_score(env, &self.lines[day - env.day], day)? * mul;

            if let Some(prev_lines) = prev_lines {
                score += Self::calc_line_score(env, &prev_lines, &self.lines[day - env.day]) * mul;
            }

            prev_lines = Some(lines);
            mul /= 10;
            day += 1;
        }

        Ok(score)
    }

    fn calc_area_score(env: &Env<'_>, lines: &[Separator], day: usize) -> Result<i64, ()> {
        let mut areas = Vec::with_capacity(env.input.n);
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
            let diff = req - area;
            score += diff.max(0) as i64 * 100;
        }

        Ok(score)
    }

    fn calc_line_score(env: &Env, prev_lines: &[Separator], lines: &[Separator]) -> i64 {
        let mut score = 0;
        let mut ptr0 = 0;
        let mut ptr1 = 0;

        let lines0 = prev_lines;
        let lines1 = lines;
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
        let lines = &self.lines[0];

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
    let mut current_score = state.calc_score(&env).unwrap();
    let mut best_solution = state.clone();
    let mut best_score = current_score;

    let mut all_iter = 0;
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
        let neigh_type = rng.gen_range(0..2);

        let mut new_state = if neigh_type == 0 {
            let day = rng.gen_range(0..state.lines.len());
            let index = rng.gen_range(0..state.lines[day].len());
            let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
            let dy = sign * 10f64.powf(rng.gen_range(0.0..3.0)).round() as i32;
            let mut new_state = state.clone();
            new_state.lines[day][index].y += dy;
            new_state
        } else if neigh_type == 1 {
            let day = rng.gen_range(0..state.lines.len());
            let index = rng.gen_range(0..state.lines[day].len());
            let new_index = rng.gen_range(0..env.widths.len());
            let new_y = rng.gen_range(1..Input::W);
            let mut new_state = state.clone();
            new_state.lines[day][index] = Separator::new(new_index, new_y);
            new_state
        } else if neigh_type == 2 {
            let day = rng.gen_range(0..state.lines.len());

            let prev_lines = if day == 0 {
                let Some(prev_lines) = &env.prev_lines else {
                    continue;
                };

                prev_lines
            } else {
                &state.lines[0]
            };

            let i0 = rng.gen_range(0..prev_lines.len());
            let i1 = rng.gen_range(0..state.lines[day].len());
            let mut new_state = state.clone();
            new_state.lines[day][i1] = prev_lines[i0];
            new_state
        } else {
            let day = rng.gen_range(0..state.lines.len());
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

            for sep in new_state.lines[day].iter_mut() {
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
