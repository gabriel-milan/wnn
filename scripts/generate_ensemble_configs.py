import json
from pathlib import Path
from copy import deepcopy

# Configurations path
CONFIGS_PATH = "ensemble_configs/"

# Parameters
ENSEMBLE_OPTIONS = [
    "majority",
    "weighted",
    "weighted_threshold",
]
N_BEST_RANGE = list(range(2, 21))
THRESHOLD_RANGE = [0.1, 0.25, 1/3, 0.5, 2/3, 0.75, 0.9, 1.0]
WEIGHTS_FACTOR = [0.1, 0.25, 1/3, 0.5, 2/3,
                  0.75, 0.9, 1.0, 1.1, 4/3, 1.5, 2, 4, 10]

# Base config
base_cfg = {
    "n_best": 2,
    "mode": "majority",
    "weights": [1, 1],
    "threshold": 0.5,
}


def generate_weights(n_best, weights_factor):
    """ Function for weights generation """
    w = [weights_factor**(i) for i in range(n_best)]
    w = [i / sum(w) for i in w]
    return w


def save_config(id: int, cfg: dict) -> None:
    """ Function for saving configurations """
    with open("{}/cfg_{}.json".format(CONFIGS_PATH, id), "w") as f:
        json.dump(cfg, f)
        f.close()


i = 0
Path(CONFIGS_PATH).mkdir(parents=True, exist_ok=True)
for n_best in N_BEST_RANGE:
    cfg = deepcopy(base_cfg)
    cfg["n_best"] = n_best
    for mode in ENSEMBLE_OPTIONS:
        cfg["mode"] = mode
        if "weighted" in mode:
            for weights_factor in WEIGHTS_FACTOR:
                cfg["weights"] = generate_weights(n_best, weights_factor)
                if "threshold" in mode:
                    for threshold in THRESHOLD_RANGE:
                        cfg["threshold"] = threshold
                        save_config(i, cfg)
                        i += 1
                else:
                    save_config(i, cfg)
                    i += 1
        else:
            save_config(i, cfg)
            i += 1
