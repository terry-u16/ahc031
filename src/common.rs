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

/// 除算を高速に行う構造体
///
/// https://nu50218.dev/posts/integer-division-by-multiplication/
#[derive(Debug, Clone, Copy)]
pub struct Divisor {
    l: u64,
    m: u64,
}

impl Divisor {
    pub const fn new(div: u32) -> Self {
        let div = div as u64;
        let l = ((div << 1) - 1).ilog2() as u64;
        let m = ((1u64 << (32 + l)) + div - 1) / div;

        Self { l, m }
    }

    pub const fn div(&self, x: u32) -> u32 {
        let mut t = x as u64;
        t *= self.m;
        t >>= 32 + self.l;
        t as u32
    }
}

pub struct DivisorSet<const N: usize> {
    dividers: [Divisor; N],
}

impl<const N: usize> DivisorSet<N> {
    pub const fn new() -> Self {
        let mut dividers = [Divisor { l: 0, m: 0 }; N];
        let mut i = 0;

        while i < dividers.len() {
            dividers[i] = Divisor::new(i as u32 + 1);
            i += 1;
        }

        Self { dividers }
    }

    pub const fn div(&self, x: u32, div: u32) -> u32 {
        self.dividers[div as usize - 1].div(x)
    }
}

pub const DIVISOR: DivisorSet<1000> = DivisorSet::new();

#[cfg(test)]
mod test {
    use rand::Rng as _;

    use super::*;

    #[test]
    fn test_divider() {
        let mut rng = rand::thread_rng();

        for _ in 0..1000 {
            let d = rng.gen_range(1..=1000);
            let div = Divisor::new(d);
            let x = rng.gen_range(0..=10000);

            assert_eq!(div.div(x), x / d as u32);
        }
    }
}
