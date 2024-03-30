use itertools::Itertools;
use rand::prelude::*;

use crate::{common::ChangeMinMax, problem::bin_packing::step1, solver::Solver};

use super::{Input, Rect};

pub struct Climber;

impl Solver for Climber {
    fn solve(&mut self, input: &Input) -> Vec<Vec<Rect>> {
        let dividers = step1::get_best_width(input);
        eprintln!("{:?}", dividers);

        let mut widths = vec![];

        for (x0, x1) in dividers.iter().tuple_windows() {
            widths.push(x1 - x0);
        }

        let mut env = Env::new(input, &dividers, widths, 0, None);
        let mut rects = vec![];
        let mut score = 0;

        for day in 0..input.days {
            env.day = day;
            let mut state = env.prev_state.clone().unwrap_or_else(|| {
                let mut heights = vec![vec![1000]; env.widths.len()];
                let div0 = input.n - env.widths.len() + 1;
                heights[0].clear();
                let mut y = 0;

                for i in 0..div0 {
                    let new_y = Input::W * (i + 1) as i32 / div0 as i32;
                    heights[0].push(new_y - y);
                    y = new_y;
                }

                State::new(heights)
            });

            for _ in 0..10 {
                eprintln!("{:?}", state);

                if state.calc_area_score(&env) == 0 {
                    break;
                }

                state = state.neigh(&env, &mut thread_rng());
            }

            let s = state.calc_score(&env);
            score += s;
            rects.push(state.to_rects(&env));
            env.prev_state = Some(state);
            eprintln!("day: {}, score: {}", day, s);
        }

        rects
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
    fn new(
        input: &'a Input,
        dividers: &'a [i32],
        widths: Vec<i32>,
        day: usize,
        prev_state: Option<State>,
    ) -> Self {
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
    heights: Vec<Vec<i32>>,
}

impl State {
    fn new(heights: Vec<Vec<i32>>) -> Self {
        Self { heights }
    }

    fn neigh(mut self, env: &Env, rng: &mut impl Rng) -> Self {
        let mut areas = Vec::with_capacity(self.heights.len());
        let mut indices = Vec::with_capacity(env.input.n);

        for (i, (heights, &width)) in self.heights.iter().zip(env.widths.iter()).enumerate() {
            let mut area = Vec::with_capacity(heights.len());

            for (j, &height) in heights.iter().enumerate() {
                area.push(height * width);
                indices.push((i, j));
            }

            areas.push(area);
        }

        let mut sorted_indices = indices.clone();

        glidesort::sort_by_key(&mut sorted_indices, |&(i, j)| areas[i][j]);
        let mut sorted_areas = areas.iter().flatten().copied().collect_vec();
        sorted_areas.sort_unstable();
        eprintln!("{:?}", sorted_areas);
        eprintln!("{:?}", env.input.requests[env.day]);

        // find bad index
        let mut bad_index = None;

        for (index, (&(i, j), &req)) in sorted_indices
            .iter()
            .zip(env.input.requests[env.day].iter())
            .enumerate()
            .rev()
        {
            let area = areas[i][j];

            if area < req {
                bad_index = Some(index);
                break;
            }
        }

        let Some(bad_index) = bad_index else {
            return self;
        };

        // try merge
        let mut best_i = !0;
        let mut best_j = !0;
        let mut best_score = i32::MAX;
        let target_req = env.input.requests[env.day][bad_index];

        for i in 0..self.heights.len() {
            let width = env.widths[i];

            for j in 0..self.heights[i].len() - 1 {
                let h0 = self.heights[i][j];
                let h1 = self.heights[i][j + 1];

                let merged_area = (h0 + h1) * width;

                let score = if merged_area >= target_req {
                    merged_area - target_req
                } else {
                    (target_req - merged_area) * 1000
                };

                if best_score.change_min(score) {
                    best_i = i;
                    best_j = j;
                }
            }
        }

        self.heights[best_i][best_j] += self.heights[best_i][best_j + 1];
        self.heights[best_i].remove(best_j + 1);

        // find split point
        let mut areas = Vec::with_capacity(self.heights.len());
        let mut indices = Vec::with_capacity(env.input.n);

        for (i, (heights, &width)) in self.heights.iter().zip(env.widths.iter()).enumerate() {
            let mut area = Vec::with_capacity(heights.len());

            for (j, &height) in heights.iter().enumerate() {
                area.push(height * width);
                indices.push((i, j));
            }

            areas.push(area);
        }

        let mut sorted_indices = indices.clone();

        glidesort::sort_by_key(&mut sorted_indices, |&(i, j)| areas[i][j]);

        let mut target_index = 0;
        let mut target_area = env.input.requests[env.day][0];

        for k in (0..env.input.n - 1).rev() {
            let (i, j) = sorted_indices[k];
            let area = areas[i][j];
            let request = env.input.requests[env.day][k + 1];

            if area < request {
                target_index = k + 1;
                target_area = request;
                break;
            }
        }

        eprintln!("{}", target_area);
        let mut split = None;

        for k in 0..env.input.n - 1 {
            // TODO: 累積和使ってちゃんとやる
            let (i, j) = sorted_indices[k];
            let request = env.input.requests[env.day][k + 1];

            let h0 = (target_area + env.widths[i] - 1) / env.widths[i];
            let a0 = h0 * env.widths[i];
            let h1 = self.heights[i][j] - h0;
            let a1 = h1 * env.widths[i];

            if a0 >= target_area && a1 >= request {
                split = Some((i, j, h0));
                eprintln!("split: {}, {}, {}", i, j, h0);
                break;
            }
        }

        let (i, j, h) = split.unwrap_or_else(|| {
            for &(i, j) in indices.iter() {
                if self.heights[i][j] >= 2 {
                    return (i, j, 1);
                }
            }

            unreachable!();
        });

        let h = if rng.gen_bool(0.5) {
            h
        } else {
            self.heights[i][j] - h
        };

        self.heights[i][j] -= h;
        self.heights[i].insert(j + 1, h);

        self
    }

    fn calc_score(&self, env: &Env) -> i64 {
        let mut score = self.calc_area_score(env);

        if let Some(prev_state) = &env.prev_state {
            score += self.calc_line_score(prev_state, env);
        }

        score
    }

    fn calc_area_score(&self, env: &Env) -> i64 {
        let mut score = 0;

        let mut areas = Vec::with_capacity(env.input.n);

        for i in 0..self.heights.len() {
            let width = env.widths[i];

            for &height in self.heights[i].iter() {
                areas.push(height * width);
            }
        }

        glidesort::sort(&mut areas);

        for (&area, &req) in areas.iter().zip(env.input.requests[env.day].iter()) {
            score += (req - area).max(0) as i64 * 100;
        }

        score
    }

    fn calc_line_score(&self, prev_state: &State, env: &Env) -> i64 {
        let mut score = 0;
        let lines0 = prev_state.get_lines();
        let lines1 = self.get_lines();

        let mut ptr0 = 0;
        let mut ptr1 = 0;
        let inf = (usize::MAX, i32::MAX);

        while ptr0 < lines0.len() || ptr1 < lines1.len() {
            let line0 = lines0.get(ptr0).copied().unwrap_or(inf);
            let line1 = lines1.get(ptr1).copied().unwrap_or(inf);

            if line0 == line1 {
                ptr0 += 1;
                ptr1 += 1;
            } else if line0 < line1 {
                score += env.widths[line0.0] as i64;
                ptr0 += 1;
            } else {
                score += env.widths[line1.0] as i64;
                ptr1 += 1;
            }
        }

        score
    }

    fn get_lines(&self) -> Vec<(usize, i32)> {
        let mut lines = vec![];

        for i in 0..self.heights.len() {
            let mut y = 0;

            for j in 0..self.heights[i].len() {
                if y != 0 {
                    lines.push((i, y));
                }

                y += self.heights[i][j];
            }
        }

        lines
    }

    fn to_rects(&self, env: &Env) -> Vec<Rect> {
        let mut rects = Vec::with_capacity(env.input.n);

        for i in 0..self.heights.len() {
            let x0 = env.dividers[i];
            let x1 = env.dividers[i + 1];
            let mut y = 0;

            for &height in self.heights[i].iter() {
                rects.push(Rect::new(x0, y, x1, y + height));
                y += height;
            }
        }

        rects.sort_unstable_by_key(|r| r.area());

        rects
    }
}
