use itertools::Itertools as _;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

use crate::{common::ChangeMinMax, problem::annealier2d, solver::Solver};

use super::{Input, Rect};

pub struct BreakAndBestFit;

impl Solver for BreakAndBestFit {
    fn solve(&mut self, input: &Input) -> Vec<Vec<Rect>> {
        let dividers = crate::problem::bin_packing::step1::get_best_width(input);
        eprintln!("{:?}", dividers);

        let rects = if dividers.len() >= 3 {
            divide(input, &dividers)
        } else {
            let mut solver = annealier2d::Annealer2d;
            solver.solve(&input)
        };
        rects
    }
}

fn divide(input: &Input, dividers: &[i32]) -> Vec<Vec<Rect>> {
    let mut rng = Pcg64Mcg::from_entropy();
    let mut prev_state = None;
    let mut rects = vec![];

    for day in 0..input.days {
        let env = Env::new(input, dividers, day, prev_state.clone());
        let mut best_state = State::new(vec![]);
        let mut best_score = i64::MAX;

        for _ in 0..1000 {
            let state = prev_state.clone().unwrap_or_else(|| State::new(vec![]));
            let mut state = state.gen_next(&env, &mut rng);

            let score = state.calc_score(&env).unwrap();

            if best_score.change_min(score) {
                best_state = state;
                eprintln!("[day {}] {}", day, score);
            }
        }

        rects.push(best_state.to_rects(&env));
        prev_state = Some(best_state);
    }

    rects
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
        glidesort::sort(&mut lines);
        Self { lines }
    }

    fn get_spaces(&self, env: &Env) -> Vec<Space> {
        let mut spaces = vec![];
        let mut pointer = 0;

        for (i, &width) in env.widths.iter().enumerate() {
            let mut y = 0;

            while pointer < self.lines.len() && self.lines[pointer].index == i {
                spaces.push(Space::new(i, width, y, self.lines[pointer].y - y));
                y = self.lines[pointer].y;

                pointer += 1;
            }

            spaces.push(Space::new(i, width, y, Input::W - y));
        }

        spaces
    }

    fn gen_next(mut self, env: &Env, rng: &mut impl Rng) -> Self {
        let take_count = rng.gen_range(0..=self.lines.len());
        let mut new_lines = self
            .lines
            .choose_multiple(rng, take_count)
            .copied()
            .collect_vec();
        new_lines.sort_unstable();

        self.lines = new_lines;
        let spaces = self.get_spaces(env);
        let mut remaining_areas = spaces.iter().map(|s| s.width * s.height).collect_vec();
        let mut indices = vec![vec![]; remaining_areas.len()];

        for (i, &req) in env.input.requests[env.day].iter().enumerate().rev() {
            let mut best_score = i64::MAX;
            let mut best_index = !0;
            let mut best_needed_area = 0;

            for (j, (space, &area)) in spaces.iter().zip(remaining_areas.iter()).enumerate() {
                let needed_y = (req + space.width - 1) / space.width;
                let needed_area = needed_y * space.width;

                let score = if area >= needed_area {
                    (area - req) as i64
                } else {
                    (needed_area - area) as i64 * 100000000
                };

                if best_score.change_min(score) {
                    best_index = j;
                    best_needed_area = needed_area;
                }
            }

            indices[best_index].push(i);
            remaining_areas[best_index] -= best_needed_area;
        }

        // lineに戻す
        let mut new_lines = vec![vec![0]; env.widths.len()];

        for (space, indices) in spaces.iter().zip(indices.iter()) {
            if indices.len() == 0 {
                new_lines[space.index].push(space.y + space.height);
                continue;
            }

            let mut y = 0;
            let mut v = vec![];

            for &index in indices.iter() {
                let req = env.input.requests[env.day][index];
                let needed_y = (req + space.width - 1) / space.width;
                y += needed_y;
                v.push(y);
            }

            let last_y = v.last().copied().unwrap();

            if last_y <= space.height {
                let remaining = space.height - last_y;

                for y in v.iter_mut() {
                    *y += remaining;
                }
            } else {
                for y in v.iter_mut() {
                    *y = *y * space.height / last_y;
                }
            }

            for y in v.iter() {
                new_lines[space.index].push(space.y + *y);
            }
        }

        // 作りすぎたlinesを削除
        let mut space_count = new_lines.iter().map(|v| v.len() - 1).sum::<usize>();

        while space_count > env.input.n {
            // 一番小さいやつを削除
            let mut area_pair = (i32::MAX, i32::MAX);
            let mut best_i = !0;
            let mut best_j = !0;

            for (i, lines) in new_lines.iter().enumerate() {
                for (j, (&y0, &y1, &y2)) in lines.iter().tuple_windows().enumerate() {
                    let h0 = y1 - y0;
                    let h1 = y2 - y1;
                    let a0 = h0 * env.widths[i];
                    let a1 = h1 * env.widths[i];

                    if area_pair.change_min((a0, a1)) {
                        best_i = i;
                        best_j = j + 1;
                    }

                    if area_pair.change_min((a1, a0)) {
                        best_i = i;
                        best_j = j + 1;
                    }
                }
            }

            new_lines[best_i].remove(best_j);
            space_count -= 1;
        }

        self.lines = new_lines
            .into_iter()
            .enumerate()
            .flat_map(|(i, lines)| lines.into_iter().map(move |y| Separator::new(i, y)))
            .filter(|s| s.y > 0 && s.y < Input::W)
            .collect_vec();
        self
    }

    fn calc_score(&mut self, env: &Env) -> Result<i64, ()> {
        glidesort::sort(&mut self.lines);
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

        glidesort::sort(&mut areas);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Space {
    index: usize,
    width: i32,
    y: i32,
    height: i32,
}

impl Space {
    fn new(index: usize, width: i32, y: i32, height: i32) -> Self {
        Self {
            index,
            width,
            y,
            height,
        }
    }
}
