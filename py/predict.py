import json
import math
import os

import numpy as np
import optuna
import pack


def gaussian_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
) -> float:
    return t1 * np.dot(x1, x2) + t2 * np.exp(-np.linalg.norm(x1 - x2) ** 2 / t3)


def calc_kernel_matrix(
    x1: np.ndarray,
    x2: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
) -> np.ndarray:
    n = x1.shape[0]
    m = x2.shape[0]
    k = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            k[i, j] = gaussian_kernel(x1[i, :], x2[j, :], t1, t2, t3)

    return k


def predict_y(
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> float:
    y_mean = np.mean(y)
    k = calc_kernel_matrix(x, x, t1, t2, t3) + t4 * np.eye(x.shape[0])
    kk = calc_kernel_matrix(x, xx, t1, t2, t3)
    yy = kk.transpose() @ np.linalg.solve(k, y - y_mean)
    return yy + y_mean


def calc_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> float:
    y_mean = np.mean(y)
    y = y - y_mean
    k = calc_kernel_matrix(x, x, t1, t2, t3) + t4 * np.eye(x.shape[0])
    yy = y.transpose() @ np.linalg.solve(k, y)
    return -np.log(np.linalg.det(k)) - yy


class Objective:
    x: np.ndarray
    y: np.ndarray

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __call__(self, trial: optuna.trial.Trial) -> float:
        t1 = trial.suggest_float("t1", 1e-6, 1e-2, log=True)
        t2 = trial.suggest_float("t2", 0.0001, 1.0, log=True)
        t3 = trial.suggest_float("t3", 0.0001, 1.0, log=True)
        t4 = trial.suggest_float("t4", 0.001, 10.0, log=True)
        return calc_log_likelihood(self.x, self.y, t1, t2, t3, t4)


def load_data():
    x_list = []
    r_list = []
    t00_list = []
    t01_list = []
    t10_list = []
    t11_list = []

    OPT_RESULT_DIR = "data/opt1"

    files = os.listdir(OPT_RESULT_DIR)

    for file in files:
        if not file.endswith(".json"):
            continue

        with open(f"{OPT_RESULT_DIR}/{file}", "r") as f:
            data = json.load(f)
            x = []
            x.append(data["d"] / 50)
            x.append(data["n"] / 50)
            x.append(math.sqrt(data["e"]) / 500)
            x_list.append(x)

            r_list.append(data["params"]["ratio"])
            t00_list.append(math.log10(data["params"]["temp00"]))
            t01_list.append(math.log10(data["params"]["temp01"]))
            t10_list.append(math.log10(data["params"]["temp10"]))
            t11_list.append(math.log10(data["params"]["temp11"]))

    x_matrix = np.array(x_list, dtype=np.float64)
    r_array = np.array(r_list, dtype=np.float64)
    t00_array = np.array(t00_list, dtype=np.float64)
    t01_array = np.array(t01_list, dtype=np.float64)
    t10_array = np.array(t10_list, dtype=np.float64)
    t11_array = np.array(t11_list, dtype=np.float64)

    return x_matrix, r_array, t00_array, t01_array, t10_array, t11_array


def predict_one(
    x_matrix: np.ndarray, data_array: np.ndarray, new_x: np.ndarray, n_trials: int = 500
) -> tuple[float, float, float, float, float]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
    )
    objective = Objective(x_matrix, data_array)
    study.optimize(objective, n_trials=n_trials)

    print("param", study.best_trial.params)

    t1 = study.best_trial.params["t1"]
    t2 = study.best_trial.params["t2"]
    t3 = study.best_trial.params["t3"]
    t4 = study.best_trial.params["t4"]

    optuna.logging.set_verbosity(optuna.logging.INFO)

    pred = predict_y(x_matrix, data_array, new_x, t1, t2, t3, t4)

    return pred, t1, t2, t3, t4


def predict(
    d: int, n: int, e: float, n_trials: int = 500
) -> tuple[float, float, float, float, float]:
    (x_matrix, r_array, t00_array, t01_array, t10_array, t11_array) = load_data()

    d_norm = d / 50
    n_norm = n / 50
    e_norm = math.sqrt(e) / 500

    new_x = np.array(
        [[d_norm, n_norm, e_norm]],
        dtype=np.float64,
    )

    pred_r, _, _, _, _ = predict_one(x_matrix, r_array, new_x, n_trials)
    pred_t00, _, _, _, _ = predict_one(x_matrix, t00_array, new_x, n_trials)
    pred_t00 = 10**pred_t00
    pred_t01, _, _, _, _ = predict_one(x_matrix, t01_array, new_x, n_trials)
    pred_t01 = 10**pred_t01
    pred_t10, _, _, _, _ = predict_one(x_matrix, t10_array, new_x, n_trials)
    pred_t10 = 10**pred_t10
    pred_t11, _, _, _, _ = predict_one(x_matrix, t11_array, new_x, n_trials)
    pred_t11 = 10**pred_t11

    return (
        pred_r[0],
        pred_t00[0],
        pred_t01[0],
        pred_t10[0],
        pred_t11[0],
    )


if __name__ == "__main__":
    (x_matrix, r_array, t00_array, t01_array, t10_array, t11_array) = load_data()

    d = 25
    n = 25
    e = 62500

    print(f"d={d}, n={n}, e={e}")

    d_norm = d / 50
    n_norm = n / 50
    e_norm = math.sqrt(e) / 500

    new_x = np.array(
        [[d_norm, n_norm, e_norm]],
        dtype=np.float64,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=== r ===")
    pred_r, t1, t2, t3, t4 = predict_one(x_matrix, r_array, new_x)
    param_r = [t1, t2, t3, t4]
    print("pred_r", pred_r)

    print("=== t00 ===")
    pred_t00, t1, t2, t3, t4 = predict_one(x_matrix, t00_array, new_x)
    pred_t00 = 10**pred_t00
    param_t00 = [t1, t2, t3, t4]
    print("pred_t00", pred_t00)

    print("=== t01 ===")
    pred_t01, t1, t2, t3, t4 = predict_one(x_matrix, t01_array, new_x)
    pred_t01 = 10**pred_t01
    param_t01 = [t1, t2, t3, t4]
    print("pred_t01", pred_t01)

    print("=== t10 ===")
    pred_t10, t1, t2, t3, t4 = predict_one(x_matrix, t10_array, new_x)
    pred_t10 = 10**pred_t10
    param_t10 = [t1, t2, t3, t4]
    print("pred_t10", pred_t10)

    print("=== t11 ===")
    pred_t11, t1, t2, t3, t4 = predict_one(x_matrix, t11_array, new_x)
    pred_t11 = 10**pred_t11
    param_t11 = [t1, t2, t3, t4]
    print("pred_t11", pred_t11)

    PARAM_PATH = "data/params.txt"

    with open(PARAM_PATH, "w") as f:
        params = {}

        params["len"] = len(x_matrix)

        d_vec = x_matrix[:, 0]
        n_vec = x_matrix[:, 1]
        e_vec = x_matrix[:, 2]

        f.write(f'const D1: &[u8] = b"{pack.pack_vec(d_vec)}";\n')
        f.write(f'const N1: &[u8] = b"{pack.pack_vec(n_vec)}";\n')
        f.write(f'const E1: &[u8] = b"{pack.pack_vec(e_vec)}";\n')

        f.write(f'const R: &[u8] = b"{pack.pack_vec(r_array)}";\n')
        f.write(f'const T00: &[u8] = b"{pack.pack_vec(t00_array)}";\n')
        f.write(f'const T01: &[u8] = b"{pack.pack_vec(t01_array)}";\n')
        f.write(f'const T10: &[u8] = b"{pack.pack_vec(t10_array)}";\n')
        f.write(f'const T11: &[u8] = b"{pack.pack_vec(t11_array)}";\n')

        f.write(
            f'const PARAM_R: &[u8] = b"{pack.pack_vec(np.array(param_r, dtype=np.float64))}";\n'
        )
        f.write(
            f'const PARAM_T00: &[u8] = b"{pack.pack_vec(np.array(param_t00, dtype=np.float64))}";\n'
        )
        f.write(
            f'const PARAM_T01: &[u8] = b"{pack.pack_vec(np.array(param_t01, dtype=np.float64))}";\n'
        )
        f.write(
            f'const PARAM_T10: &[u8] = b"{pack.pack_vec(np.array(param_t10, dtype=np.float64))}";\n'
        )
        f.write(
            f'const PARAM_T11: &[u8] = b"{pack.pack_vec(np.array(param_t11, dtype=np.float64))}";\n'
        )
