use nalgebra::{DMatrix, DVector, DVectorView};

// 初回パラメータ調整分
const D1: &[u8] = b"mpmZmZmZ6T9cj8L1KFzvP7gehetRuO4/mpmZmZmZyT+kcD0K16PgP7gehetRuM4/uB6F61G4zj9mZmZmZmbmP3E9CtejcN0/cT0K16Nw3T/NzMzMzMzsP4XrUbgehes/9ihcj8L16D+kcD0K16PgPylcj8L1KNw/AAAAAAAA4D9xPQrXo3DtPxSuR+F6FO4/mpmZmZmZuT8pXI/C9SjcP5qZmZmZmck/MzMzMzMz4z+F61G4HoXrP+xRuB6F6+E/AAAAAAAA4D8fhetRuB7lP2ZmZmZmZuY/FK5H4XoU7j+4HoXrUbjuP+xRuB6F68E/uB6F61G47j/sUbgehevhP5qZmZmZmbk/KVyPwvUozD+4HoXrUbi+Pw==";
const N1: &[u8] = b"j8L1KFyP4j9xPQrXo3DdPwAAAAAAAPA/KVyPwvUo7D/D9Shcj8LVPwAAAAAAAPA/rkfhehSu5z9xPQrXo3DdP1K4HoXrUeg/mpmZmZmZ2T9mZmZmZmbmP/YoXI/C9eg/j8L1KFyP4j+kcD0K16PgP5qZmZmZmck/mpmZmZmZyT9mZmZmZmbmPylcj8L1KNw/KVyPwvUozD8pXI/C9SjMP65H4XoUruc/pHA9Ctej4D/sUbgehevhP3sUrkfhetQ/pHA9Ctej4D+F61G4HoXrP/YoXI/C9eg/AAAAAAAA8D97FK5H4XrUP+F6FK5H4eo/uB6F61G43j/sUbgehevRP9ejcD0K1+M/rkfhehSu5z/sUbgehevhPw==";
const E1: &[u8] = b"eCisVdsk4j9Ke2rbUWviPwJkIvc0PO0/CKwcWmQ73z85UilnmTPiP6224FT6wug/Mwc3QH0uvz+vkiOmDybbPx9gFxi8PLs/fyC6PKsT5D8Gvc5YlJDrP5huEoPAyuk/b9UVzFDK0j+zNpwhRki/Pw7qk9lFSss/EPegqd/SzT+ereKYgcDmPy6SuAFnROk/bzFCuBAxyD82noegDXruP24KuIzIB+U/YsNyjLLR2D/d7I4rMVPFP/xR4k67ytk/Kghh7/k/xD/p85IJm/bePy12ef7qXcs/zqIra7aWwD/s03VE5gTcP96V0DSBL+g/F9nO91Pj1T/IlTsKMonZP3B5TBma6NA/SyzgD1uP7j9/Jb7Eey/UPw==";
const R: &[u8] = b"WgBLlj4Czj/k2gg3Zq/RP95Z2GksVtE/4mfNlHMZ0j/wFRQQ0RbGP28mFg2tl9I/hT3PIt4e3T+/TH80nbzWPyzoobNCVck/NYeJKt+w2z+Y40CGRTnRPznJT1IbuuI/NWdCiUPA0z8BLtk6D+/UP5CpR5vXrd4/QZJowglm1T+pJ7jHeIrLP5qZmZmZmbk/yO9DiEPI4D8cGvAbA2/QP4CBs4Vwk9M/mpmZmZmZuT/qTZRYn2PSPweHK+ZTJsI/zGJo8LYn2j+rOL8NsazFP2884peLBN4/ccu0z/7K1z9W01YwVtKpP2VlgERN4uE/7KHOM7x01T+amZmZmZm5P8PD0ca5Nt4/l53A1u/2sj9FoSyPvUjfPw==";
const T00: &[u8] = b"1JC8bIu1HEBzELOkV9gZQDGBFlEPQxhA7D0Zwcg0FkCd1gewBf0YQND84gN2yRZALG6fibIoFkD+0FBfhK0UQEJD7h6HnBVAYbQZWr5wFkBcy5iy+1oUQMWNVSLnqRRASYijnAzpGEB3Un5/dNMYQMSQRmGt9R9A259AoQZYG0A7LQYVFsAZQAAAAAAAABxAKxwnuLlWFED7St+Nl8gYQE+6PqbLrxdAAAAAAAAAHEBcw/oNDhQXQEx78pzBax9AUDEyyuknF0Ac0zC1IqsXQGnihu1uUBdAGqYv6EHmFkDSIRtoV6kdQAHALi6f5htAYY+73W5EF0AAAAAAAAAcQHdFwZQE4RhA4UUCZwLdHECf1VCfwSsXQA==";
const T01: &[u8] = b"Rf+OzENPxL8+NkeEb9Kevw8OMp9iAtU/cL3lecJo0b/kWUQJ8vDpPwbaai+R2Ok/2tOKRqQ89z9U34IoUKnXv1qApgbTA90/+vhP3Aki7z8fGXuNx4DfP1pLuo4T7+Y/VT7lMVzZ9j/ykdRXAerhPzfotz+s6eM/BtgPa4Lj2T9cLKNaHAXRvwAAAAAAAAAAf1WOF+sU8j8o315xEhzUP4zLB5lLvek/AAAAAAAAAADIXiIcekzzP3KzOChhCJi/Pk7GNixX7D82V6rVkZzrP7i9KuJXROs/fxELUVLM17//tKevbFfvPyQn/N7NsvI/N9VXXc/Xrr8AAAAAAAAAAFK3w/qn+PY/zs7/umUVyj+M767K2u7hPw==";
const T10: &[u8] = b"YNuZfawBBEBtNaVZfdIHQCoaYI9qXP4/7SUiW8WwA0A3aXxb3BoEQIM28hymAv0/j+uZbymSC0DGW0872z8BQJIOx+a7FQtA1XraNTwTAkD1T6M3Gg8GQPalddexvQFAhpkOJ8H0/z81X7XmEloLQDN4EDfUfAtA0mDJUB+mB0A1jSa8pOsEQEDvE2qCaAJAu1kaLFioCUCwYtWjpH0CQCbG85POZ/0/QO8TaoJoAkDgQqCCSLD4P+to/kBc+QVA2ZgZYvoOB0CjpCK5Nlb/P/gWLlrUmP0/aFiVO5Tc/z8HwSxaWnsGQBX1HJAPswZAkjg7Mq0L+D9A7xNqgmgCQPV2bDxDmgJA/V9Qxf2vAEBObVOiowkAQA==";
const T11: &[u8] = b"u7RpwyNQ17+byHERSFidvzyQ4F2he+C/yZzbbprX0j8UBoBZOcPvP8JDU2fmVeC/mkjba0U29z923GhJk5PpP2H9fYhM6+8/IrRi8+BB8T8EU9gbXTqwPx4KbdR66dw/YSBQ1y5Z5j9s2Ireocf0P7NGj1QjIPY/kPMNUgUq5D9Yl5LEOvbfP/3VT5Ynid4/Zv0L4HZT1j/MWu3yG+n2P1SP+98tNps//dVPlieJ3j/S33hYO2fsP42wkx7bZb4/lv47xGjq8D8S3WSu76m4P+M1C3+urqK/sXNz8knG6T+KCTjZ9LDsP91LHSG9/p0/UreVb6EY7j/91U+WJ4neP7HIHfySOeU/eRAlcgqcwr/uvYoN4GntPw==";
const PARAM_R: &[u8] = b"FfCuGCAR/D4ukq2Qo9+SP07YczbDp3s/9hohCeTtUj8=";
const PARAM_T00: &[u8] = b"orNE4KgkOD+FubF9d5/cP9sPHw9VRXo/DkUyGhhYvD8=";
const PARAM_T01: &[u8] = b"ImZNmTVEyD79XIYnl/nTP5kfssGNppc/YxDybVBigD8=";
const PARAM_T10: &[u8] = b"lUWOw/LiDD+D2YFRpnTMP1GvfepS+78/yvwDCx7IuT8=";
const PARAM_T11: &[u8] = b"8rBnEActyD6bhnL/bmPKP7Hkggq2s9w/bqihl2x+wz8=";

pub struct ParamSuggester {
    x_matrix: DMatrix<f64>,
    y_vector: DVector<f64>,
    hyper_param: DVector<f64>,
    y_inv_trans: fn(f64) -> f64,
    lower: f64,
    upper: f64,
}

impl ParamSuggester {
    fn new(
        hyper_param: DVector<f64>,
        x_matrix: DMatrix<f64>,
        y_vector: DVector<f64>,
        y_inv_trans: fn(f64) -> f64,
        lower: f64,
        upper: f64,
    ) -> Self {
        Self {
            hyper_param,
            x_matrix,
            y_vector,
            y_inv_trans,
            lower,
            upper,
        }
    }

    fn gen_x_matrix_1() -> DMatrix<f64> {
        let d = DVector::from_vec(decode_base64(D1)).transpose();
        let n = DVector::from_vec(decode_base64(N1)).transpose();
        let e = DVector::from_vec(decode_base64(E1)).transpose();

        let x_matrix = DMatrix::from_rows(&[d, n, e]);

        x_matrix
    }

    pub fn gen_ratio_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_R));
        let y_vector = DVector::from_vec(decode_base64(R));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| x,
            0.05,
            0.8,
        )
    }

    pub fn gen_t00_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_T00));
        let y_vector = DVector::from_vec(decode_base64(T00));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| 10.0f64.powf(x),
            1e5,
            1e8,
        )
    }

    pub fn gen_t01_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_T01));
        let y_vector = DVector::from_vec(decode_base64(T01));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| 10.0f64.powf(x),
            1e-1,
            1e2,
        )
    }

    pub fn gen_t10_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_T10));
        let y_vector = DVector::from_vec(decode_base64(T10));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| 10.0f64.powf(x),
            1e1,
            1e4,
        )
    }

    pub fn gen_t11_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_T11));
        let y_vector = DVector::from_vec(decode_base64(T11));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| 10.0f64.powf(x),
            1e-1,
            1e2,
        )
    }

    pub fn suggest(&self, d: usize, n: usize, e: f64) -> f64 {
        let d = d as f64 / 50.0;
        let n = n as f64 / 50.0;
        let e = e.sqrt() / 500.0;
        let len = self.x_matrix.shape().1;
        let y_mean = self.y_vector.mean();
        let y_mean = DVector::from_element(self.y_vector.len(), y_mean);
        let new_x = DMatrix::from_vec(3, 1, vec![d, n, e]);
        let noise = DMatrix::from_diagonal_element(len, len, self.hyper_param[3]);

        let k = self.calc_kernel_matrix(&self.x_matrix, &self.x_matrix) + noise;
        let kk = self.calc_kernel_matrix(&self.x_matrix, &new_x);

        let kernel_lu = k.lu();
        let new_y = kk.transpose() * kernel_lu.solve(&(&self.y_vector - &y_mean)).unwrap();

        (self.y_inv_trans)(new_y[(0, 0)] + y_mean[(0, 0)]).clamp(self.lower, self.upper)
    }

    fn calc_kernel_matrix(&self, x1: &DMatrix<f64>, x2: &DMatrix<f64>) -> DMatrix<f64> {
        let n = x1.shape().1;
        let m = x2.shape().1;
        let mut kernel = DMatrix::zeros(n, m);

        for i in 0..n {
            for j in 0..m {
                kernel[(i, j)] = self.gaussian_kernel(&x1.column(i), &x2.column(j));
            }
        }

        kernel
    }

    fn gaussian_kernel(&self, x1: &DVectorView<f64>, x2: &DVectorView<f64>) -> f64 {
        let t1 = self.hyper_param[0];
        let t2 = self.hyper_param[1];
        let t3 = self.hyper_param[2];

        let diff = x1 - x2;
        let norm_diff = diff.dot(&diff);
        let dot = x1.dot(&x2);
        t1 * dot + t2 * (-norm_diff / t3).exp()
    }
}

fn decode_base64(data: &[u8]) -> Vec<f64> {
    const BASE64_MAP: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut stream = vec![];

    let mut cursor = 0;

    while cursor + 4 <= data.len() {
        let mut buffer = 0u32;

        for i in 0..4 {
            let c = data[cursor + i];
            let shift = 6 * (3 - i);

            for (i, &d) in BASE64_MAP.iter().enumerate() {
                if c == d {
                    buffer |= (i as u32) << shift;
                }
            }
        }

        for i in 0..3 {
            let shift = 8 * (2 - i);
            let value = (buffer >> shift) as u8;
            stream.push(value);
        }

        cursor += 4;
    }

    let mut result = vec![];
    cursor = 0;

    while cursor + 8 <= stream.len() {
        let p = stream.as_ptr() as *const f64;
        let x = unsafe { *p.offset(cursor as isize / 8) };
        result.push(x);
        cursor += 8;
    }

    result
}
