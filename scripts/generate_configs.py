import json
from pathlib import Path
from copy import deepcopy

# Defaults
DEFAULT_RANDOM_SEED = 8080
DEFAULT_TRAIN_VALIDATION_SPLIT = 0.2
DEFAULT_TRAIN_FOLDS = 10
DEFAULT_WSD_IGNORE_ZERO = False
DEFAULT_WSD_VERBOSE = False

# Parameters
BINARIZATION_OPTIONS = [
    "threshold",
    "thermometer",
    "circular_thermometer",
    "sauvola",
    "niblack",
    "adaptive_thresh_gaussian",
    "adaptive_thresh_mean"
]
BINARIZATION_THRESHOLDS = [-192, -128, -64, -32, 0, 32, 64, 128, 192]
BINARIZATION_RESOLUTIONS = [8, 16, 24, 32, 48, 64]
NS_WINDOW = [3, 5, 7, 9, 11]
ADDRESS_SIZES = [15, 16, 17, 18, 19,
                 20, 21, 22, 23, 24,
                 25, 26, 27, 28, 29,
                 30, 31, 32, 33, 34, ]
FEATURE_SET_OPTIONS = [
    "rings",
    "shower_shape"
]
WSD_CLUSTER = [True, False]
CONSTANT_C = [1, 2, 3, 4]
CONSTANT_K = [0.2, 0.4, 0.6, 0.8]

CLUS_THRESHOLD = [1, 2, 3, 5, 10, 15, 20, 30]
CLUS_DISCRIMINATOR_LIMIT = [1, 2, 3, 5, 10, 15, 20, 30]
CLUS_MIN_SCORE = [0.1, 0.2, 0.3, 0.4, 0.5]

base_cfg = {
    "wsd_cluster": False,
    "random_seed": DEFAULT_RANDOM_SEED,
    "train_validation_split": DEFAULT_TRAIN_VALIDATION_SPLIT,
    "train_folds": DEFAULT_TRAIN_FOLDS,
    "binarization": "",
    "binarization_threshold": 0,
    "binarization_resolution": 0,
    "wsd_address_size": 0,
    "wsd_ignore_zero": False,
    "wsd_verbose": False,
    "clus_min_score": 0.1,
    "clus_threshold": 0,
    "clus_discriminator_limit": 0,
    "window_size": 3,
    "constant_c": 2,
    "constant_k": 0.2,
}


def save_config(id: int, cfg: dict) -> None:
    with open("configs/cfg_{}.json".format(id), "w") as f:
        json.dump(cfg, f)
        f.close()


i = 0
Path("configs/").mkdir(parents=True, exist_ok=True)
for feature_set in FEATURE_SET_OPTIONS:
    cfg = deepcopy(base_cfg)
    cfg["feature_set"] = feature_set
    for binarization_option in BINARIZATION_OPTIONS:
        cfg["binarization"] = binarization_option
        if binarization_option == "threshold":
            for binarization_threshold in BINARIZATION_THRESHOLDS:
                cfg["binarization_threshold"] = binarization_threshold
                for address_size in ADDRESS_SIZES:
                    cfg["wsd_address_size"] = address_size
                    for wsd_clus in WSD_CLUSTER:
                        cfg["wsd_cluster"] = wsd_clus
                        if wsd_clus:
                            for clus_t in CLUS_THRESHOLD:
                                cfg["clus_threshold"] = clus_t
                                for clus_d in CLUS_DISCRIMINATOR_LIMIT:
                                    cfg["clus_discriminator_limit"] = clus_d
                                    for clus_m in CLUS_MIN_SCORE:
                                        cfg["clus_min_score"] = clus_m
                                        save_config(i, cfg)
                                        i += 1
                        else:
                            save_config(i, cfg)
                            i += 1
        elif binarization_option in ["thermometer", "circular_thermometer"]:
            for binarization_resolution in BINARIZATION_RESOLUTIONS:
                cfg["binarization_resolution"] = binarization_resolution
                for address_size in ADDRESS_SIZES:
                    cfg["wsd_address_size"] = address_size
                    for wsd_clus in WSD_CLUSTER:
                        cfg["wsd_cluster"] = wsd_clus
                        if wsd_clus:
                            for clus_t in CLUS_THRESHOLD:
                                cfg["clus_threshold"] = clus_t
                                for clus_d in CLUS_DISCRIMINATOR_LIMIT:
                                    cfg["clus_discriminator_limit"] = clus_d
                                    for clus_m in CLUS_MIN_SCORE:
                                        cfg["clus_min_score"] = clus_m
                                        save_config(i, cfg)
                                        i += 1
                        else:
                            save_config(i, cfg)
                            i += 1
        elif binarization_option == "sauvola":
            for window_size in NS_WINDOW:
                cfg["window_size"] = window_size
                for address_size in ADDRESS_SIZES:
                    cfg["wsd_address_size"] = address_size
                    for wsd_clus in WSD_CLUSTER:
                        cfg["wsd_cluster"] = wsd_clus
                        if wsd_clus:
                            for clus_t in CLUS_THRESHOLD:
                                cfg["clus_threshold"] = clus_t
                                for clus_d in CLUS_DISCRIMINATOR_LIMIT:
                                    cfg["clus_discriminator_limit"] = clus_d
                                    for clus_m in CLUS_MIN_SCORE:
                                        cfg["clus_min_score"] = clus_m
                                        save_config(i, cfg)
                                        i += 1
                        else:
                            save_config(i, cfg)
                            i += 1
        elif binarization_option == "niblack":
            for window_size in NS_WINDOW:
                cfg["window_size"] = window_size
                for constant_k in CONSTANT_K:
                    cfg["constant_k"] = constant_k
                    for address_size in ADDRESS_SIZES:
                        cfg["wsd_address_size"] = address_size
                        for wsd_clus in WSD_CLUSTER:
                            cfg["wsd_cluster"] = wsd_clus
                            if wsd_clus:
                                for clus_t in CLUS_THRESHOLD:
                                    cfg["clus_threshold"] = clus_t
                                    for clus_d in CLUS_DISCRIMINATOR_LIMIT:
                                        cfg["clus_discriminator_limit"] = clus_d
                                        for clus_m in CLUS_MIN_SCORE:
                                            cfg["clus_min_score"] = clus_m
                                            save_config(i, cfg)
                                            i += 1
                            else:
                                save_config(i, cfg)
                                i += 1
        elif binarization_option in ["adaptive_thresh_mean", "adaptive_thresh_gaussian"]:
            for window_size in NS_WINDOW:
                cfg["window_size"] = window_size
                for address_size in ADDRESS_SIZES:
                    cfg["wsd_address_size"] = address_size
                    for constant_c in CONSTANT_C:
                        cfg["constant_c"] = constant_c
                        for wsd_clus in WSD_CLUSTER:
                            cfg["wsd_cluster"] = wsd_clus
                            if wsd_clus:
                                for clus_t in CLUS_THRESHOLD:
                                    cfg["clus_threshold"] = clus_t
                                    for clus_d in CLUS_DISCRIMINATOR_LIMIT:
                                        cfg["clus_discriminator_limit"] = clus_d
                                        for clus_m in CLUS_MIN_SCORE:
                                            cfg["clus_min_score"] = clus_m
                                            save_config(i, cfg)
                                            i += 1
                            else:
                                save_config(i, cfg)
                                i += 1
