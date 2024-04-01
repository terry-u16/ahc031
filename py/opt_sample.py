import datetime
import json
import math
import os
import random
import shutil
import subprocess

import optimize
import optuna

subprocess.run("cargo build --release", shell=True)
shutil.move("./target/release/ahc031", "./ahc031")

SEED_PATH = "data/seed.txt"
INPUT_PATH = "data/in"
OPT_PATH = "data/opt1"
# os.environ["DURATION_MUL"] = "1.5"
# os.environ["AHC030_SHOW_COMMENT"] = "0"

with open(SEED_PATH, "w") as f:
    for seed in range(0, 1000):
        f.write(f"{seed}\n")

for iteration in range(1, 1000):
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"ahc031-{start_time}-group-{iteration:000}"

    W = 1000
    d = random.randint(5, 50)
    n = random.randint(5, 50)
    e = random.randint(500, 5000) / 10000
    E = int(round(W * W * e * e))

    cmd = f"./gen {SEED_PATH} -d {INPUT_PATH} --D {d} --N {n} --E {E}"
    print(cmd)

    subprocess.run(cmd, shell=True).check_returncode()

    objective = optimize.Objective()

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )

    ratio = 0.1
    temp00 = 1e7
    temp01 = 1e0
    temp10 = 2e2
    temp11 = 3e0

    print(
        f"suggested params: r={ratio}, t00={temp00}, t01={temp01}, t10={temp10}, t11={temp11}"
    )

    study.enqueue_trial(
        {
            "ratio": ratio,
            "temp00": temp00,
            "temp01": temp01,
            "temp10": temp10,
            "temp11": temp11,
        }
    )

    study.optimize(objective, timeout=120)

    dictionary = {
        "study_name": study_name,
        "d": d,
        "n": n,
        "e": E,
        "params": study.best_trial.params,
    }

    filename = (
        "optimized_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    with open(f"{OPT_PATH}/{filename}", "w") as f:
        json.dump(dictionary, f, indent=2)

    # optuna.visualization.plot_param_importances(study).show()
    # optuna.visualization.plot_contour(study).show()
