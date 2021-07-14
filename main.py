from db import *
from utils import *

import json
from sys import argv
import wisardpkg as wp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split


class Gym:
    """Class for training and getting results"""

    def __init__(self, config_file: str):

        # Load configuration file
        try:
            self._config: dict = None
            with open(config_file, "r") as f:
                self._config = json.load(f)
                f.close()
        except Exception as e:
            print(f"Failed to load configuration file: {e}")

        # Load data
        try:
            self.X_ss, self.X_ring, self.y = load_data()
            if self._config["feature_set"] == "rings":
                self.X = self.X_ring
            elif self._config["feature_set"] == "shower_shape":
                self.X = self.X_ss
            else:
                raise Exception(
                    f'Feature set {self._config["feature_set"]} unknown')
        except Exception as e:
            print(f"Failed to load data: {e}")

        # Binarization
        try:
            binarizer = Binarizer()
            self.X_binary = {}
            if self._config["binarization"] == "threshold":
                for i in range(5):
                    self.X_binary[i] = binarizer.basic_bin(
                        self.X[i], threshold=self._config["binarization_threshold"])
            elif self._config["binarization"] == "thermometer":
                for i in range(5):
                    self.X_binary[i] = binarizer.simple_thermometer(
                        self.X[i], resolution=self._config["binarization_resolution"])
            elif self._config["binarization"] == "circular_thermometer":
                for i in range(5):
                    self.X_binary[i] = binarizer.circular_thermometer(
                        self.X[i], resolution=self._config["binarization_resolution"])
            elif self._config["binarization"] == "sauvola":
                for i in range(5):
                    self.X_binary[i] = binarizer.sauvola(
                        self.X[i], window_size=self._config["window_size"])
            elif self._config["binarization"] == "niblack":
                for i in range(5):
                    self.X_binary[i] = binarizer.niblack(
                        self.X[i], window_size=self._config["window_size"],
                        k=self._config["constant_k"])
            elif self._config["binarization"] == "adaptive_thresh_mean":
                for i in range(5):
                    self.X_binary[i] = binarizer.adaptive_thresh_mean(
                        self.X[i], window_size=self._config["window_size"],
                        constant_c=self._config["constant_c"])
            elif self._config["binarization"] == "adaptive_thresh_gaussian":
                for i in range(5):
                    self.X_binary[i] = binarizer.adaptive_thresh_gaussian(
                        self.X[i], window_size=self._config["window_size"],
                        constant_c=self._config["constant_c"])
            else:
                raise Exception(
                    f'Binarization type {self._config["binarization"]} unknown.')
        except Exception as e:
            print(f"Fail on data binarization: {e}")

        # Model
        self._model: list = [None for _ in range(5)]
        self._trained: list = [False for _ in range(5)]

    @property
    def trained(self) -> bool:
        return all(self._trained)

    @property
    def config(self) -> dict:
        return self._config

    def train(self, X_train: list, y_train: list, i: int = 0) -> None:
        """Trains a brand new Wisard model"""

        # Reset trained flag
        self._trained[i] = False

        # Build Wisard model
        try:
            self._model[i] = wp.Wisard(
                self._config["wsd_address_size"],
                ignoreZero=self._config["wsd_ignore_zero"],
                verbose=self._config["wsd_verbose"]
            )
        except Exception as e:
            print(f"Fail on building wisard model: {e}")

        # Train model
        self._model[i].train(X_train, y_train)

        # Set trained flag
        self._trained[i] = True

    def train_cluster(self, X_train: list, y_train: list, i: int = 0) -> None:
        """Trains a brand new Wisard Cluster model"""

        # Reset trained flag
        self._trained[i] = False

        # Build Wisard model
        try:
            address_size = self._config["wsd_address_size"]
            min_score = self._config["clus_min_score"]
            threshold = self._config["clus_threshold"]
            discriminator_limit = self._config["clus_discriminator_limit"]
            self._model[i] = wp.ClusWisard(
                address_size, min_score, threshold, discriminator_limit)
        except Exception as e:
            print(f"Fail on building wisard cluster model: {e}")

        # Train model
        self._model[i].train(X_train, y_train)

        # Set trained flag
        self._trained[i] = True

    def predict(self, X: list, i: int = 0) -> list:
        """Makes classification for model"""
        if not self._trained[i]:
            raise Exception("Model needs training before evaluation!")
        return self._model[i].classify(X)

    def evaluate(self, X: list, y: list, i: int = 0) -> float:
        """Makes classification for model and returns accuracy"""
        y_pred: list = self.predict(X, i=i)
        return accuracy_score(y, y_pred)  # TODO: Change to SP metric

    def train_full(self) -> None:
        """Trains using full train set"""
        for i in range(5):
            if self._config.get('wsd_cluster'):
                self.train_cluster(self.X_binary[i], self.y[i], i=i)
            else:
                self.train(self.X_binary[i], self.y[i], i=i)

    def train_with_split(self, validation_split: float = None) -> list:
        """Trains using `validation_split` percentage of train set 
        for validation. Returns accuracy for the validation set"""

        scores = [None for _ in range(5)]
        for i in range(5):
            # Split into train and validation
            validation_split = validation_split if validation_split else self._config[
                "train_validation_split"]
            X_train, X_valid, y_train, y_valid = train_test_split(
                self.X_binary[i], self.y[i], test_size=validation_split, random_state=self._config["random_seed"])

            # Train model
            if self._config.get('wsd_cluster'):
                self.train_cluster(X_train, y_train, i=i)
            else:
                self.train(X_train, y_train, i=i)

            # Evaluate model and return
            scores[i] = self.evaluate(X_valid, y_valid, i=i)

        return scores

    def train_with_kfold(self, folds: int = None, verbose: bool = False) -> list:
        """Trains using `folds` for the number of K-folds. 
        Returns average accuracy for validation sets."""

        folds = folds if folds else self._config["train_folds"]
        kf = StratifiedKFold(n_splits=folds, shuffle=True,
                             random_state=self._config["random_seed"])
        ret_scores = [None for _ in range(5)]
        for j in range(5):
            if verbose:
                print("#" * 25)
                print(f"## ET {j + 1}")
                print("#" * 25)
            scores = []
            for fold, (idxT, idxV) in enumerate(kf.split(self.X_binary[j], self.y[j])):
                if verbose:
                    print(f"- FOLD {fold + 1}: ", end="")

                X_train = [self.X_binary[j][i] for i in idxT]
                X_valid = [self.X_binary[j][i] for i in idxV]
                y_train = [self.y[j][i] for i in idxT]
                y_valid = [self.y[j][i] for i in idxV]

                if self._config.get('wsd_cluster'):
                    self.train_cluster(X_train, y_train)
                else:
                    self.train(X_train, y_train)
                score: float = self.evaluate(X_valid, y_valid)
                if verbose:
                    print("Score: {:.4f}%".format(score*100))
                scores.append(score)

            ret_scores[j] = np.mean(scores)
        return ret_scores


if __name__ == "__main__":
    try:
        config_file_path: str = argv[1]
    except IndexError:
        raise IndexError("Please provide the path to the configuration file")
    db = DBManager()
    g = Gym(config_file_path)
    val_acc = g.train_with_kfold()
    test_acc = g.evaluate(g.X_test_binary, g.y_test)
    db.add_train_result(g.config, val_accuracy=val_acc, test_accuracy=test_acc)
