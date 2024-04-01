use crate::{
    common::ChangeMinMax,
    problem::{Input, Rect},
};

use super::Solver;

pub struct ScoreOne;

impl Solver for ScoreOne {
    fn solve(&self, input: &Input) -> (Vec<Vec<Rect>>, i64) {
        let mut areas = vec![];

        for i in 0..input.n {
            let mut max = 0;

            for day in 0..input.days {
                max.change_max(input.requests[day][i]);
            }

            areas.push(max);
        }

        let sum = areas.iter().sum::<i32>();

        if sum > Input::W * Input::W {
            return (vec![], i64::MAX);
        }

        match dp(input, &areas) {
            Some(rects) => (vec![rects; input.days], 1),
            None => (vec![], i64::MAX),
        }
    }
}

fn dp(input: &Input, areas: &[i32]) -> Option<Vec<Rect>> {
    let mut prefix_sum = vec![0; input.n + 1];

    for i in 0..input.n {
        prefix_sum[i + 1] = prefix_sum[i] + areas[i];
    }

    let mut dp = vec![i32::MAX / 2; input.n + 1];
    let mut from = vec![!0; input.n + 1];
    dp[0] = 0;

    for i in 0..input.n {
        for j in i + 1..=input.n {
            let mut ok = Input::W;
            let mut ng = 0;

            while ok - ng > 1 {
                let mid = (ok + ng) / 2;
                let height_set = get_height_set(&areas[i..j], mid);
                let height_sum = height_set.iter().sum::<i32>();

                if height_sum <= Input::W {
                    ok = mid;
                } else {
                    ng = mid;
                }
            }

            let new_score = dp[i] + ok;

            if dp[j].change_min(new_score) {
                from[j] = i;
            }
        }
    }

    if dp[input.n] <= Input::W {
        let mut current = input.n;
        let mut rects = vec![];

        while current > 0 {
            let prev = from[current];
            let x0 = dp[prev];
            let x1 = dp[current];
            let heights = get_height_set(&areas[prev..current], x1 - x0);
            let mut y = 0;

            for &h in heights.iter() {
                let next_y = y + h;
                rects.push(Rect::new(x0, y, x1, next_y));
                y = next_y;
            }

            current = prev;
        }

        rects.sort_unstable_by_key(|r| r.area());
        Some(rects)
    } else {
        None
    }
}

fn get_height_set(areas: &[i32], width: i32) -> Vec<i32> {
    let mut height_set = vec![];

    for &area in areas.iter() {
        height_set.push((area + width - 1) / width);
    }

    height_set
}
