import json
from os import environ
from pathlib import Path
from copy import deepcopy

from db import DBManager, TrainTable

N = 15
BEST_CONFIGS_DIR = "best_configs/"
ENV_FILE = ".env"

# Create best configs dir if it does not exist
Path(BEST_CONFIGS_DIR).mkdir(parents=True, exist_ok=True)

# Loads env file
with open(ENV_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("#"):
            continue
        key, value = line.split("=")
        environ[key] = value

# Get session from DBManager
session = DBManager().session

# Query for the N best models
best_models = session.query(TrainTable).order_by((TrainTable.et1_val_sp + TrainTable.et2_val_sp +
                                                  TrainTable.et3_val_sp + TrainTable.et4_val_sp + TrainTable.et5_val_sp).desc()).limit(N).all()

# Get base config from file
with open("run_config.json", "r") as f:
    base_config = json.load(f)

# Write configs to file
for i, model in enumerate(best_models):
    with open(BEST_CONFIGS_DIR + str(i) + ".json", "w") as f:
        cfg = deepcopy(base_config)
        cfg["wsd_cluster"] = model.wsd_cluster
        cfg["feature_set"] = model.feature_set
        cfg["binarization"] = model.binarization
        cfg["binarization_threshold"] = model.binarization_threshold
        cfg["binarization_resolution"] = model.binarization_resolution
        cfg["wsd_address_size"] = model.wsd_address_size
        cfg["wsd_ignore_zero"] = model.wsd_ignore_zero
        cfg["clus_min_score"] = model.clus_min_score
        cfg["clus_threshold"] = model.clus_threshold
        cfg["clus_discriminator_limit"] = model.clus_discriminator_limit
        cfg["window_size"] = model.window_size
        cfg["constant_c"] = model.constant_c
        cfg["constant_k"] = model.constant_k
        json.dump(cfg, f, indent=4)
