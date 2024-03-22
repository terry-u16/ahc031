use itertools::Itertools;
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Input {
    days: usize,
    n: usize,
    requests: Vec<Vec<i64>>,
}

impl Input {
    const W: i64 = 1000;

    fn read() -> Self {
        input! {
            _w: i64,
            days: usize,
            n: usize,
            requests: [[i64; n]; days],
        }

        Self { days, n, requests }
    }
}

fn main() {
    let input = Input::read();
    let state = State::init(&input);

    let state = annealing(&input, state, 2.9);
    state.print(&input);
}

#[derive(Debug, Clone)]
struct State {
    lines: Vec<Vec<Option<i64>>>,
}

impl State {
    fn new(lines: Vec<Vec<Option<i64>>>) -> Self {
        Self { lines }
    }

    fn init(input: &Input) -> Self {
        let mut lines = vec![];

        for day in 0..input.days {
            let mut prefix_sum = vec![0];

            for a in input.requests[day].iter() {
                prefix_sum.push(prefix_sum.last().unwrap() + a);
            }

            let sum = prefix_sum.last().unwrap();

            let mut l = vec![];

            for (i, s) in prefix_sum.iter().enumerate() {
                l.push(Some(((s * Input::W + sum / 2) / sum).max(i as i64)));
            }

            lines.push(l);
        }

        Self { lines }
    }

    fn calc_score(&self, input: &Input) -> Option<i64> {
        let mut score = 1;

        let mut prev_lines = vec![!0; input.n + 1];
        let mut prev_lines_sorted = vec![!0; input.n + 1];
        let mut lines = vec![];
        let mut areas = vec![];

        for (day, (requests, xs)) in input.requests.iter().zip(self.lines.iter()).enumerate() {
            lines.clear();
            areas.clear();

            for i in 0..=input.n {
                let x = xs[i];
                let prev = prev_lines[i];

                let x = match x {
                    Some(x) => x,
                    None => prev,
                };

                prev_lines[i] = x;
                lines.push(x);
            }

            glidesort::sort(&mut lines);

            let mut i0 = 0;
            let mut i1 = 0;

            if day != 0 {
                while i0 < input.n || i1 < input.n {
                    let l0 = lines[i0];
                    let l1 = prev_lines_sorted[i1];

                    if l0 == l1 {
                        i0 += 1;
                        i1 += 1;
                    } else if l0 < l1 {
                        i0 += 1;
                        score += Input::W;
                    } else {
                        i1 += 1;
                        score += Input::W;
                    }
                }
            }

            prev_lines_sorted.copy_from_slice(&lines);

            for (x0, x1) in lines.iter().tuple_windows() {
                let dx = x1 - x0;

                if dx == 0 {
                    return None;
                }

                areas.push(dx * Input::W);
            }

            glidesort::sort(&mut areas);

            for (&request, &area) in requests.iter().zip(areas.iter()) {
                let diff = area - request;

                if diff < 0 {
                    score -= diff * 100;
                }
            }
        }

        Some(score)
    }

    fn print(&self, input: &Input) {
        let mut prev_lines = vec![!0; input.n + 1];
        let mut lines = vec![];
        let mut pairs = vec![];

        for xs in self.lines.iter() {
            lines.clear();
            pairs.clear();

            for i in 0..=input.n {
                let x = xs[i];
                let prev = prev_lines[i];

                let x = match x {
                    Some(x) => x,
                    None => prev,
                };

                prev_lines[i] = x;
                lines.push(x);
            }

            glidesort::sort(&mut lines);

            for (&x0, &x1) in lines.iter().tuple_windows() {
                pairs.push((x0, x1));
            }

            glidesort::sort_by_key(&mut pairs, |(x0, x1)| x1 - x0);

            for (x0, x1) in pairs.iter() {
                println!("{} {} {} {}", x0, 0, x1, Input::W);
            }
        }
    }
}

fn annealing(input: &Input, initial_solution: State, duration: f64) -> State {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_score(input).unwrap();
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 1e7;
    let temp1 = 3e2;
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
        let day = rng.gen_range(0..input.days);
        let i = rng.gen_range(1..input.n);

        let same_pos = rng.gen_bool(0.2);

        if same_pos && (day == 0 || solution.lines[day][i].is_none()) {
            continue;
        }

        let mut new_solution = solution.clone();

        if same_pos {
            new_solution.lines[day][i] = None;
        } else {
            let old_x = match solution.lines[day][i] {
                Some(x) => x,
                None => {
                    let mut d = day - 1;

                    loop {
                        if let Some(x) = solution.lines[d][i] {
                            break x;
                        }

                        d -= 1;
                    }
                }
            };

            let x = loop {
                let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
                let dx = sign * 10.0f64.powf(rng.gen_range(0.0..3.0)).round() as i64;
                let x = old_x + dx;

                if 0 < x && x < Input::W {
                    break x;
                }
            };

            new_solution.lines[day][i] = Some(x);
        }

        // スコア計算
        let Some(new_score) = new_solution.calc_score(input) else {
            continue;
        };
        let score_diff = new_score - current_score;

        if score_diff <= 0 || rng.gen_bool(f64::exp(-score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;
            solution = new_solution;

            if best_score.change_min(current_score) {
                best_solution = solution.clone();
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

    best_solution
}
