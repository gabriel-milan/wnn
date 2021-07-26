from typing import List, Any

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from main import Gym
from utils import sp


class WeightedThresholdVotingEnsembler:

    def __init__(self, models: List[Gym], weights: List[float], threshold: float):
        try:
            assert len(models) == len(weights)
        except AssertionError:
            raise ValueError("Number of models and weights must be the same")
        self.models = models
        self.weights = weights
        self.weight_sum = sum(weights)
        self.threshold = threshold

    def get_scores(self, ensembled_preds: List[List[List[str]]]) -> list:
        ret_scores = [None for _ in range(5)]
        for et_bin in range(len(ensembled_preds)):
            kf = StratifiedKFold(n_splits=len(ensembled_preds[et_bin]), shuffle=True,
                                 random_state=self.models[0].config["random_seed"])
            scores = []
            for fold, (_, idxV) in enumerate(kf.split(self.models[0].X_binary[et_bin], self.models[0].y[et_bin])):
                score: list = sp([self.models[0].y[et_bin][i] for i in idxV],
                                 ensembled_preds[et_bin][fold], return_pd_fa=True)
                scores.append(np.array(score))
            ret_scores[et_bin] = np.mean(np.array(scores), axis=0)
        return ret_scores

    def ensemble_preds(self, preds: List[List[List[Any]]]) -> List[List[List[str]]]:
        ensemble_preds = np.zeros((len(preds[0]), len(preds[0][0]))).tolist()
        for et_bin in tqdm(range(len(preds[0]))):
            for fold_idx in range(len(preds[0][et_bin])):
                fold_preds = np.zeros((len(preds[0][et_bin][fold_idx])))
                for model_idx in range(len(preds)):
                    fold_preds += np.array(preds[model_idx][et_bin][fold_idx]) * \
                        self.weights[model_idx] / self.weight_sum
                ensemble_preds[et_bin][fold_idx] = [
                    str(int(x >= self.threshold)) for x in fold_preds]
        return ensemble_preds

    def evaluate_with_kfold(self) -> List[np.ndarray]:
        preds = []
        for model in tqdm(self.models):
            _, model_preds = model.train_with_kfold()
            preds.append(model_preds)
        ensembled_preds = self.ensemble_preds(preds)
        scores = self.get_scores(ensembled_preds)
        return scores


class WeightedVotingEnsembler(WeightedThresholdVotingEnsembler):
    def __init__(self, models: List[Gym], weights: List[float], threshold: float = None):
        super().__init__(models, weights, 0.5)


class MajorityVotingEnsembler (WeightedVotingEnsembler):
    def __init__(self, models: List[Gym], weights: List[float] = None, threshold: float = None):
        super().__init__(models, [1 for _ in models], 0.5)
