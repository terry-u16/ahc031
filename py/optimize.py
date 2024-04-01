import json
import math
import re
import shutil
import subprocess

import optuna

TIME_RATIO = 1.5


class Objective:
    def __init__(self) -> None:
        pass

    def __call__(self, trial: optuna.trial.Trial) -> float:
        ratio = trial.suggest_float("ratio", 0.05, 0.6, log=False)
        temp00 = trial.suggest_float("temp00", 1e5, 1e8, log=True)
        temp01 = trial.suggest_float("temp01", 3e-1, 3e1, log=True)
        temp10 = trial.suggest_float("temp10", 3e1, 3e3, log=True)
        temp11 = trial.suggest_float("temp11", 3e-1, 3e1, log=True)

        min_seed = 0
        max_seed = 539
        batch_size = 180
        step = 0
        score_sum = 0.0
        args = f"{ratio} {temp00} {temp01} {temp10} {temp11}"
        local_execution = f"ahc031 {args}"
        cloud_execution = f"ahc031 {args}"
        print(f">> {local_execution}")

        for begin in range(min_seed, max_seed + 1, batch_size):
            end = begin + batch_size

            with open("runner_config_original.json", "r") as f:
                config = json.load(f)

            config["RunnerOption"]["StartSeed"] = begin
            config["RunnerOption"]["EndSeed"] = end
            config["ExecutionOption"]["LocalExecutionSteps"][0][
                "ExecutionCommand"
            ] = local_execution
            config["ExecutionOption"]["CloudExecutionSteps"][0][
                "ExecutionCommand"
            ] = cloud_execution

            with open("runner_config.json", "w") as f:
                json.dump(config, f, indent=2)

            command = "dotnet marathon run-local"
            process = subprocess.run(command, stdout=subprocess.PIPE, encoding="utf-8", shell=True)

            lines = process.stdout.splitlines()
            score_pattern = r"rate:\s*(\d+.\d+)%"

            for line in lines:
                result = re.search(score_pattern, line)
                if result:
                    score = float(result.group(1))
                    if score > 0.0:
                        score = math.log10(score)
                    else:
                        score = math.log10(10.00)
                    score_sum += score

            if end < max_seed + 1:
                trial.report(score_sum, step)
                print(f"{score_sum:.5f}", end=" ", flush=True)

                if trial.should_prune():
                    print()
                    raise optuna.TrialPruned()

            step += 1

        print(f"{score_sum:.5f}")
        return score_sum


if __name__ == "__main__":
    STUDY_NAME = "ahc031-000"

    # subprocess.run("dotnet marathon compile-rust")
    subprocess.run("cargo build --release", shell=True)
    shutil.move("./target/release/ahc031", "./ahc031")

    objective = Objective()

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )

    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "ratio": 0.1,
                "temp00": 1e7,
                "temp01": 1e0,
                "temp10": 2e2,
                "temp11": 3e0,
            }
        )

    study.optimize(objective, n_trials=20)
    print(study.best_trial)

    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_slice(study, params=["r", "k"]).show()
    optuna.visualization.plot_contour(study).show()
