import json
from sys import argv
from pathlib import Path

from tqdm import tqdm

from main import Gym
from db import DBManager
from ensemblers import MajorityVotingEnsembler, WeightedVotingEnsembler, WeightedThresholdVotingEnsembler

BEST_CONFIGS_DIR = "best_configs/"

# Create best configs dir if it does not exist
configs_path = Path(BEST_CONFIGS_DIR)

if __name__ == "__main__":
    try:
        config_file_path: str = argv[1]
    except IndexError:
        raise IndexError("Please provide the path to the configuration file")

    with open(config_file_path, "r") as f:
        config = json.load(f)

    if config["mode"] == "majority":
        Ensembler = MajorityVotingEnsembler
    elif config["mode"] == "weighted":
        Ensembler = WeightedVotingEnsembler
    elif config["mode"] == "weighted_threshold":
        Ensembler = WeightedThresholdVotingEnsembler

    models = []
    for i in tqdm(range(config["n_best"])):
        models.append(Gym(configs_path / f"{i}.json"))

    ensembler = Ensembler(models, config["weights"], config["threshold"])
    db = DBManager()
    scores = ensembler.evaluate_with_kfold()
    db.add_ensemble_result(config, scores)
