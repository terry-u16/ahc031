use itertools::Itertools as _;
use rand::Rng as _;

use crate::{common::ChangeMinMax as _, solver::Solver};

use super::{Input, Rect};

pub struct Annealer2d;

impl Solver for Annealer2d {
    fn solve(&mut self, input: &Input) -> (Vec<Vec<Rect>>, i64) {
        let sizes = dp(&input);
        let (coords, indices) = build_squares(&sizes);
        let env = Env::new(input.clone(), indices.clone());
        let state = State::new(vec![coords; input.days]);
        let duration = 1.45 - input.since.elapsed().as_secs_f64();
        let state = annealing(&env, state, duration);
        let score = state.calc_score(&env).unwrap();

        eprintln!("{}", score);

        let mut result = vec![];

        for coords in &state.coords {
            let mut rects = indices
                .rects
                .iter()
                .map(|r| coords.get_rect(r))
                .collect_vec();
            rects.sort_by_key(|r| r.area());

            result.push(rects);
        }

        (result, score)
    }
}

#[derive(Debug, Clone)]
struct Env {
    input: Input,
    coord_indices: CoordIndices,
}

impl Env {
    fn new(input: Input, coord_indices: CoordIndices) -> Self {
        Self {
            input,
            coord_indices,
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    coords: Vec<LineCoords>,
}

impl State {
    fn new(coords: Vec<LineCoords>) -> Self {
        Self { coords }
    }

    fn calc_score(&self, env: &Env) -> Result<i64, ()> {
        let mut score = 1;
        let mut areas = vec![];

        for (reqs, coord) in env.input.requests.iter().zip(self.coords.iter()) {
            areas.clear();

            for r in env.coord_indices.rects.iter() {
                let rect = coord.get_rect(r);
                if !rect.is_valid() {
                    return Err(());
                }

                let area = rect.area();
                areas.push(area);
            }

            glidesort::sort(&mut areas);

            for (req, area) in reqs.iter().zip(&areas) {
                score += 100 * (req - area).max(0) as i64;
            }
        }

        // 線分が偶然重なることを考慮していないので、厳密には正しくない
        for l in env.coord_indices.lines.iter() {
            let mut l0 = self.coords[0].get_line(l);

            for coord in self.coords.iter().skip(1) {
                let l1 = coord.get_line(l);
                score += l0.diff(&l1) as i64;
                l0 = l1;
            }
        }

        Ok(score)
    }

    fn calc_score_day(&self, env: &Env, day: usize) -> Result<i64, ()> {
        let mut score = 0;
        let mut areas = vec![];
        let coord = &self.coords[day];

        for r in env.coord_indices.rects.iter() {
            let rect = coord.get_rect(r);
            if !rect.is_valid() {
                return Err(());
            }

            let area = rect.area();
            areas.push(area);
        }

        glidesort::sort(&mut areas);

        for (req, area) in env.input.requests[day].iter().zip(&areas) {
            score += 100 * (req - area).max(0) as i64;
        }

        let since = day.saturating_sub(1);
        let until = (day + 1).min(env.input.days - 1);

        // 線分が偶然重なることを考慮していないので、厳密には正しくない
        for l in env.coord_indices.lines.iter() {
            let mut l0 = self.coords[since].get_line(l);

            for coord in self.coords[since + 1..=until].iter() {
                let l1 = coord.get_line(l);
                score += l0.diff(&l1) as i64;
                l0 = l1;
            }
        }

        Ok(score)
    }
}

#[derive(Debug, Clone)]
struct LineCoords {
    coords: Vec<i32>,
}

impl LineCoords {
    fn new() -> Self {
        Self { coords: vec![] }
    }

    fn get_line(&self, line: &LineIndex) -> Line {
        Line::new(
            self.coords[line.mu0],
            self.coords[line.mu1],
            self.coords[line.xi],
        )
    }

    fn get_rect(&self, rect: &RectIndex) -> Rect {
        Rect::new(
            self.coords[rect.x0],
            self.coords[rect.y0],
            self.coords[rect.x1],
            self.coords[rect.y1],
        )
    }
}

#[derive(Debug, Clone)]
struct CoordIndices {
    lines: Vec<LineIndex>,
    rects: Vec<RectIndex>,
}

impl CoordIndices {
    fn new() -> Self {
        Self {
            lines: vec![],
            rects: vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Line {
    mu0: i32,
    mu1: i32,
    xi: i32,
}

impl Line {
    fn new(mu0: i32, mu1: i32, xi: i32) -> Self {
        Self { mu0, mu1, xi }
    }

    fn len(&self) -> i32 {
        self.mu1 - self.mu0
    }

    fn overlap(&self, other: &Self) -> i32 {
        if self.xi != other.xi {
            return 0;
        }

        let mu0 = self.mu0.max(other.mu0);
        let mu1 = self.mu1.min(other.mu1);
        (mu1 - mu0).max(0)
    }

    fn diff(&self, other: &Self) -> i32 {
        self.len() + other.len() - 2 * self.overlap(other)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LineIndex {
    mu0: usize,
    mu1: usize,
    xi: usize,
}

impl LineIndex {
    fn new(mu0: usize, mu1: usize, xi: usize) -> Self {
        Self { mu0, mu1, xi }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RectIndex {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl RectIndex {
    fn new(x0: usize, y0: usize, x1: usize, y1: usize) -> Self {
        Self { x0, y0, x1, y1 }
    }
}

fn dp(input: &Input) -> Vec<i32> {
    const INF: i32 = i32::MAX / 2;
    const MAX_POW: usize = 10;
    const SUM: usize = 1 << MAX_POW;
    let mut dp = vec![vec![INF; SUM + 1]; input.n + 1];
    let mut from = vec![vec![!0; SUM + 1]; input.n + 1];
    let sizes = (0..=MAX_POW)
        .map(|i| (1 << i) * Input::W * Input::W / (1 << MAX_POW))
        .collect_vec();
    dp[0][0] = 0;

    for i in 0..input.n {
        let target = input.requests.iter().map(|r| r[i]).max().unwrap();
        //let target = input.requests.iter().map(|r| r[i]).sum::<i64>() / input.days as i64;

        for j in 0..=SUM {
            for pow in 0..=MAX_POW {
                let next = j + (1 << pow);

                if next > SUM {
                    break;
                }

                let next_score = dp[i][j] + (target - sizes[pow]).max(0);

                if dp[i + 1][next].change_min(next_score) {
                    from[i + 1][next] = j;
                }
            }
        }
    }

    let mut sizes = vec![];
    let mut i = input.n;
    let mut j = SUM;

    while i > 0 {
        let prev = from[i][j];
        sizes.push((j - prev) as i32);
        j = prev;
        i -= 1;
    }

    sizes.reverse();
    eprintln!("best: {}", dp[input.n][SUM]);
    eprintln!("{:?}", sizes);

    sizes
}

fn build_squares(sizes: &[i32]) -> (LineCoords, CoordIndices) {
    let mut index = 0;
    let mut line_coords = LineCoords::new();
    let mut coord_indices = CoordIndices::new();
    line_coords.coords.push(0);
    line_coords.coords.push(0);
    line_coords.coords.push(Input::W);
    line_coords.coords.push(Input::W);
    coord_indices.lines.push(LineIndex::new(0, 2, 1));
    coord_indices.lines.push(LineIndex::new(0, 2, 3));
    coord_indices.lines.push(LineIndex::new(1, 3, 0));
    coord_indices.lines.push(LineIndex::new(1, 3, 2));

    sq_dfs(
        0,
        1,
        2,
        3,
        &mut index,
        1024,
        sizes,
        true,
        &mut line_coords,
        &mut coord_indices,
    );

    (line_coords, coord_indices)
}

fn sq_dfs(
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    index: &mut usize,
    size: i32,
    sizes: &[i32],
    vertical: bool,
    cd: &mut LineCoords,
    idx: &mut CoordIndices,
) {
    if sizes[*index] == size {
        idx.rects.push(RectIndex::new(x0, y0, x1, y1));
        *index += 1;
        return;
    }

    let size = size / 2;

    if vertical {
        let xx0 = cd.coords[x0];
        let xx1 = cd.coords[x1];
        let mid = (xx0 + xx1) / 2;
        let i = cd.coords.len();
        cd.coords.push(mid as i32);
        idx.lines.push(LineIndex::new(y0, y1, i));

        sq_dfs(x0, y0, i, y1, index, size, sizes, !vertical, cd, idx);
        sq_dfs(i, y0, x1, y1, index, size, sizes, !vertical, cd, idx);
    } else {
        let yy0 = cd.coords[y0];
        let yy1 = cd.coords[y1];
        let mid = (yy0 + yy1) / 2;
        let i = cd.coords.len();
        cd.coords.push(mid as i32);
        idx.lines.push(LineIndex::new(x0, x1, i));

        sq_dfs(x0, y0, x1, i, index, size, sizes, !vertical, cd, idx);
        sq_dfs(x0, i, x1, y1, index, size, sizes, !vertical, cd, idx);
    }
}

fn annealing(env: &Env, mut state: State, duration: f64) -> State {
    let mut best_solution = state.clone();
    let mut current_score = state.calc_score(&env).unwrap();
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e6;
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
        let (day, index, x) = if rng.gen_bool(0.5) {
            // copy
            let pre = rng.gen_bool(0.5);
            let mut day = rng.gen_range(0..env.input.days - 1);
            if pre {
                day += 1;
            }

            let index = rng.gen_range(4..state.coords[day].coords.len());

            let x0 = if pre {
                state.coords[day - 1].coords[index]
            } else {
                state.coords[day + 1].coords[index]
            };

            let x1 = state.coords[day].coords[index];

            if x0 == x1 {
                continue;
            }

            let ratio = rng.gen_range(0.1f64..1.5).min(1.0);
            let new_x = (x0 as f64 * ratio + x1 as f64 * (1.0 - ratio)).round() as i32;
            (day, index, new_x)
        } else {
            // shift
            let day = rng.gen_range(0..env.input.days);
            let index = rng.gen_range(4..state.coords[day].coords.len());
            let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
            let dx = sign * 10f64.powf(rng.gen_range(0.0..2.0)).round() as i32;
            let new_x = state.coords[day].coords[index] + dx;

            if new_x <= 0 || new_x >= Input::W {
                continue;
            }

            (day, index, new_x)
        };

        // スコア計算
        let prev_score_day = state.calc_score_day(&env, day).unwrap();
        let prev_x = state.coords[day].coords[index];
        state.coords[day].coords[index] = x;

        let Ok(new_score_day) = state.calc_score_day(&env, day) else {
            state.coords[day].coords[index] = prev_x;
            continue;
        };
        let score_diff = new_score_day - prev_score_day;

        if score_diff <= 0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score += score_diff;
            accepted_count += 1;

            if best_score.change_min(current_score) {
                best_solution = state.clone();
                update_count += 1;
            }
        } else {
            state.coords[day].coords[index] = prev_x;
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
