from typing import List

import os
import sys
import json

import optuna
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    train_test_split,
    KFold,
)
from sklearn.metrics import roc_auc_score


RANDOM_STATE = 0
FUSION_METHOD = sys.argv[1]  # early or late
MODEL_DATA_STR = sys.argv[2]  # specify model name

# Number of folds in each hyperoptimization trial
# Adding more folds increases duration of each trial but 
# better estimates hyperparameter performance
NUM_KFOLD_INNER = 5 

# Number of trials that are performed using randomly selected
# hyperparameter sets. Higher values promote exploration and 
# improve the starting point for the directed optimization, but 
# may not add much beyond a certain point.
NUM_STARTUP_TRIALS = (
    25  
)

# Total number of hyperoptimization trials to be performed in each fold
NUM_TRIALS = 100  

# Column name containing the target label
TARGET_COLNAME = "cs_pca" 

BEST_PARAMS = {
    # Add best parameters here and set DO_OPTIMIZATION to False to run experiment with fixed parameters
}

# Do hyperparameter optimization, or use BEST_PARAMS?
DO_OPTIMIZATION = True

DL_LIK_FEATURES = [
    "lesion_0_lkhd",
    "lesion_1_lkhd",
    "lesion_2_lkhd",
]
DL_VOL_FEATURES = [
    "lesion_0_size",
    "lesion_1_size",
    "lesion_2_size",
]
CLIN_FEATURES = [
    "psa",
    "volume",
    "PatientAge",
]

if MODEL_DATA_STR == "ai_with_clinical":
    dl_feature_cols = DL_LIK_FEATURES
    dl_vol_feature_cols = DL_VOL_FEATURES
    clin_feature_cols = CLIN_FEATURES
if MODEL_DATA_STR == "baseline":
    dl_feature_cols = [DL_LIK_FEATURES[0]]
    dl_vol_feature_cols = []
    clin_feature_cols = []
if MODEL_DATA_STR == "baseline_multi":
    dl_feature_cols = DL_LIK_FEATURES
    dl_vol_feature_cols = []
    clin_feature_cols = []
if MODEL_DATA_STR == "dl_volumetric":
    dl_feature_cols = DL_LIK_FEATURES
    dl_vol_feature_cols = DL_VOL_FEATURES
    clin_feature_cols = []
if MODEL_DATA_STR == "clinical":
    dl_feature_cols = []
    clin_feature_cols = CLIN_FEATURES
    dl_vol_feature_cols = []
if MODEL_DATA_STR == "dl_clinical":
    dl_feature_cols = DL_LIK_FEATURES
    clin_feature_cols = CLIN_FEATURES
    dl_vol_feature_cols = []

feature_cols = dl_feature_cols + clin_feature_cols + dl_vol_feature_cols

# DL likelihood features are already in 0-1 range
EXCLUDE_FROM_NORMALIZATION = DL_LIK_FEATURES
COLS_TO_NORM = [col for col in feature_cols if col not in EXCLUDE_FROM_NORMALIZATION]

RESULTS_DIR = "./prediction_results/"

def fix_decimals(df: pd.DataFrame, colname: str):
    """Ensure that both comma's and full-stops are interpreted as decimal separators"""
    try:
        df[colname] = df[colname].str.replace(",", ".", regex=True)
    except AttributeError:
        pass
    df[colname] = pd.to_numeric(df[colname])
    return df

def remove_missing(df: pd.DataFrame, colnames: List[str]):
    """Remove records with selected columns missing data"""
    df_copy = df.copy()  # Create a copy to avoid modifying the original DataFrame
    df_copy.dropna(subset=colnames, inplace=True)
    return df_copy

INTERNAL_DATASET_PATH = "./internal_dataset.csv" ### <- specify your data path here
EXTERNAL_DATASET_PATH = "./external_dataset.csv" ### <- specify your data path here

print("> Reading internal dataset..")
internal_data = pd.read_csv(INTERNAL_DATASET_PATH, sep=";")

print("> Reading external dataset..")
external_data = pd.read_csv(EXTERNAL_DATASET_PATH, sep=";")

# Prevent decimal conversion errors (e.g. from opening in Excel) in any numeric columns
internal_data = fix_decimals(internal_data, "psa")
external_data = fix_decimals(external_data, "psa")

# Deal with incomplete data
missing_data_strategy = "remove"
if missing_data_strategy in ["remove", "delete", "del", "rm"]:
    predrop_internal = len(internal_data)
    predrop_external = len(external_data)

    internal_data = remove_missing(internal_data, ["volume", "psa"])
    external_data = remove_missing(external_data, ["volume", "psa"])

    num_dropped_internal = predrop_internal - len(internal_data)
    num_dropped_external = predrop_external - len(external_data)

    print(
        "> Removed",
        num_dropped_internal,
        "records with missing values from internal data.",
    )
    print(
        "> Removed",
        num_dropped_external,
        "records with missing values from external data.",
    )

print("internal exams:", len(internal_data))
print("internal patients:", len(internal_data.PatientID.unique()))
print("external exams:", len(external_data))
print("external patients:", len(external_data.PatientID.unique()))

class Classifier:
    """
    Class that implements various machine learning classifiers, that can 
    be specified in the `classifier_type` argument.
    Allows specification of hyperparameter search space and additional fixed
    parameters.

    Additional models can be added by implementing `available_models()`, and 
    `_set_model_type()`

    Check `optuna` documentation for information about specifying search space.
    """
    def __init__(self, classifier_type="linear_svm", seed=88):
        self.classifier_type = classifier_type
        self.hyper_params = {}
        self.fixed_params = {}
        self.seed = seed
        self._model = None

        self._set_model_type(classifier_type)

    @staticmethod
    def available_models():
        return ["linear_svm", "rbf_svm", "dec_tree", "grad_boost", "mlp", "logistic"]

    def _set_model_type(self, classifier_type):
        if classifier_type == "logistic":
            self._inst_func = LogisticRegression
            self._search_space = [
                ("lr_class_weight", "choice", [None, "balanced"]),
            ]
            self.fixed_params = {}
        elif classifier_type == "linear_svm":
            self._inst_func = SVC
            self._search_space = [
                ("lsvm_C", "float", 0.1, 2.0),
            ]
            self.fixed_params = {"fx_probability": True}
        elif classifier_type == "rbf_svm":
            self._inst_func = SVC
            self._search_space = [
                ("rbf_C", "float", 0.1, 2.0),
                ("rbf_class_weight", "choice", [None, "balanced"]),
                ("rbf_gamma", "choice", ["scale", "auto"]),
            ]
            self.fixed_params = {"fx_kernel": "rbf", "fx_probability": True}
        elif classifier_type == "dec_tree":
            self._inst_func = DecisionTreeClassifier
            self._search_space = [
                ("dt_criterion", "choice", ["gini", "entropy", "log_loss"]),
                ("dt_splitter", "choice", ["best", "random"]),
                ("dt_max_depth", "int", 1, 6),
                ("dt_class_weight", "choice", [None, "balanced"]),
            ]
            self.fixed_params = {}
        elif classifier_type == "grad_boost":
            self._inst_func = GradientBoostingClassifier
            self._search_space = [
                ("gb_loss", "choice", ["log_loss", "exponential"]),
                ("gb_learning_rate", "log_float", 0.001, 0.9),
                ("gb_n_estimators", "int", 1, 10),
                ("gb_subsample", "float", 0.5, 1.0),
                ("gb_criterion", "choice", ["friedman_mse", "squared_error"]),
            ]
            self.fixed_params = {}
        elif classifier_type == "mlp":
            self._inst_func = MLPClassifier
            self._search_space = [
                ("mlp_n_layers", "int", 1, 4),
                ("mlp_n_hidden", "int", 3, 20),
                ("mlp_solver", "choice", ["adam", "sgd"]),
                ("mlp_alpha", "log_float", 0.0001, 0.001),
                ("mlp_learning_rate_init", "log_float", 1e-4, 1e-2),
                ("mlp_max_iter", "int", 50, 300),
            ]
            self.fixed_params = {"fx_activation": "logistic"}

    def suggest_hyperparameters(self, trial: optuna.Trial):
        for name, t, *args in self._search_space:
            param_name = "_".join(name.split("_")[1:])
            if t == "int":
                self.hyper_params[param_name] = trial.suggest_int(name, *args)
            if t == "float":
                self.hyper_params[param_name] = trial.suggest_float(name, *args)
            if t == "log_float":
                self.hyper_params[param_name] = trial.suggest_float(
                    name, *args, log=True
                )
            if t == "choice":
                self.hyper_params[param_name] = trial.suggest_categorical(name, *args)

    def _instantiate_model(self):
        args = {**self.hyper_params}

        for name in self.fixed_params:
            param_name = "_".join(name.split("_")[1:])
            args[param_name] = self.fixed_params[name]
        args["random_state"] = self.seed

        # Fix how the n_layers and n_hidden arguments are interpreted
        # This allows hyperopting architecture
        if self.classifier_type == "mlp":
            n_layers = args["n_layers"]
            n_hidden = args["n_hidden"]
            del args["n_layers"]
            del args["n_hidden"]

            args["hidden_layer_sizes"] = (n_hidden,) * n_layers

        self._model = self._inst_func(**args)

    def train(self, x, y):
        self._instantiate_model()
        self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict_proba(x)[..., 1]

    def __str__(self):
        _str = f"[{self.classifier_type} model (seed: {self.seed})]\n"
        _str += "++ Fixed params:\n"
        _str += json.dumps(self.fixed_params, indent=2)
        _str += "\n++ Optimized params:\n"
        _str += json.dumps(self.fixed_params, indent=2)
        _str += f"\n== ID: {super().__str__()}"
        return _str


feature_sets = {}
if FUSION_METHOD == "late":
    # Create separate models for DL and clin features
    if len(dl_feature_cols) >= 1:
        feature_sets["dl"] = dl_feature_cols
    if len(clin_feature_cols) >= 1:
        feature_sets["clin"] = clin_feature_cols
    if len(dl_vol_feature_cols) >= 1:
        feature_sets["vol"] = dl_vol_feature_cols

elif FUSION_METHOD == "early":
    # Create a single model for all features
    feature_sets["all"] = feature_cols

# Create a classifier instance
cls = Classifier()
print(cls)

# Split the data into development and testing data
development_idxs, test_idxs = train_test_split(
    range(len(internal_data)), test_size=0.2, random_state=RANDOM_STATE
)

development_data = internal_data.iloc[development_idxs]
test_data = internal_data.iloc[test_idxs]

print("development exams:", len(development_data))
print("development patients:", len(development_data.PatientID.unique()))
print("internal test exams:", len(test_data))
print("internal test patients:", len(test_data.PatientID.unique()))

all_internal_preds = {}
all_external_preds = {}
all_internal_aucs = {}
all_external_aucs = {}

print(
    "Training model for",
    MODEL_DATA_STR,
    "containing feature sets:",
    list(feature_sets.keys()),
)

for feature_set_name in feature_sets:
    print(f"Training submodel '{feature_set_name}'")
    feature_cols = feature_sets[feature_set_name]

    def objective(trial: optuna.Trial):
        # First suggest the type of classifier model we want to use, and instantiate it
        classifier_type = trial.suggest_categorical(
            "classifier_type", Classifier.available_models()
        )
        cls = Classifier(classifier_type, seed=RANDOM_STATE)

        # Fill in the necessary hyperparameters for this model:
        cls.suggest_hyperparameters(trial)

        # Create an inner cross-validation loop that splits the development data into train / valid
        inner_cv_aucs = []
        inner_kfold = KFold(
            n_splits=NUM_KFOLD_INNER, shuffle=True, random_state=RANDOM_STATE + 1
        )
        for inner_fold_num, (train_idxs, valid_idxs) in enumerate(
            inner_kfold.split(development_data)
        ):
            train_data = development_data.iloc[train_idxs]
            valid_data = development_data.iloc[valid_idxs]

            train_x = train_data[feature_cols].copy()
            train_y = train_data[TARGET_COLNAME].values
            valid_x = valid_data[feature_cols].copy()
            valid_y = valid_data[TARGET_COLNAME].values

            cols_to_norm = [col for col in COLS_TO_NORM if col in train_x.columns]
            if len(cols_to_norm) >= 1:
                means = train_x[cols_to_norm].mean(axis=0)
                stds = train_x[cols_to_norm].std(axis=0)
                train_x.loc[:, cols_to_norm] = (
                    train_x[cols_to_norm].copy() - means
                ) / stds
                valid_x.loc[:, cols_to_norm] = (
                    valid_x[cols_to_norm].copy() - means
                ) / stds

            cls.train(train_x, train_y)
            val_preds = cls.predict(valid_x)
            val_auc = roc_auc_score(valid_y, val_preds)
            inner_cv_aucs.append(val_auc)

        mean_auc = sum(inner_cv_aucs) / NUM_KFOLD_INNER
        print(f"Mean AUC: {mean_auc:.3f}")
        return mean_auc

    if DO_OPTIMIZATION:
        # Optimize the hyperparameters using Optuna for the highest AUC, and store them in best_params
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=NUM_STARTUP_TRIALS, seed=RANDOM_STATE
        )
        study = optuna.create_study(sampler=sampler, direction="maximize")

        study.optimize(objective, NUM_TRIALS)
        best_params = study.best_params
    else:
        # If no optimization is desired, use the parameters set in BEST_PARAMS
        best_params = BEST_PARAMS[feature_set_name].copy()

    print("best_params", best_params)

    # Retrain the model using the full development set
    development_x = development_data[feature_cols].copy()
    development_y = development_data[TARGET_COLNAME].values
    test_x = test_data[feature_cols].copy()
    test_y = test_data[TARGET_COLNAME].values

    cols_to_norm = [col for col in COLS_TO_NORM if col in development_x.columns]
    if len(cols_to_norm) >= 1:
        means = development_x[cols_to_norm].mean(axis=0)
        stds = development_x[cols_to_norm].std(axis=0)
        development_x.loc[:, cols_to_norm] = (
            development_x[cols_to_norm].copy() - means
        ) / stds
        test_x.loc[:, cols_to_norm] = (test_x[cols_to_norm].copy() - means) / stds

    cls = Classifier(best_params["classifier_type"], seed=RANDOM_STATE)
    del best_params["classifier_type"]

    cls.fixed_params = {**cls.fixed_params, **best_params}
    cls.train(development_x, development_y)

    internal_test_preds = cls.predict(test_x)
    internal_test_auc = roc_auc_score(test_y, internal_test_preds)

    # Retrain the model using full internal dataset
    internal_x = internal_data[feature_cols].copy()
    internal_y = internal_data[TARGET_COLNAME].values
    external_x = external_data[feature_cols].copy()
    external_y = external_data[TARGET_COLNAME].values

    cols_to_norm = [col for col in COLS_TO_NORM if col in internal_x.columns]
    if len(cols_to_norm) >= 1:
        means = internal_x[cols_to_norm].mean(axis=0)
        stds = internal_x[cols_to_norm].std(axis=0)
        internal_x.loc[:, cols_to_norm] = (
            internal_x[cols_to_norm].copy() - means
        ) / stds
        external_x.loc[:, cols_to_norm] = (
            external_x[cols_to_norm].copy() - means
        ) / stds

    cls.train(internal_x, internal_y)
    external_test_preds = cls.predict(external_x)
    external_test_auc = roc_auc_score(external_y, external_test_preds)

    all_internal_aucs[feature_set_name] = internal_test_auc
    all_external_aucs[feature_set_name] = external_test_auc
    all_internal_preds[feature_set_name] = internal_test_preds
    all_external_preds[feature_set_name] = external_test_preds

for feature_set_name in feature_sets:
    print(f"Internal AUC ({feature_set_name}):", all_internal_aucs[feature_set_name])
    print(f"External AUC ({feature_set_name}):", all_external_aucs[feature_set_name])

print("Aggregating predictions over submodels for late fusion..")
mean_internal_preds = np.mean(
    np.array(list(all_internal_preds.values())), axis=0
).tolist()
mean_external_preds = np.mean(
    np.array(list(all_external_preds.values())), axis=0
).tolist()

aggregated_internal_auc = roc_auc_score(test_y, mean_internal_preds)
aggregated_external_auc = roc_auc_score(external_y, mean_external_preds)
print(f"Aggregated internal AUC:", aggregated_internal_auc)
print(f"Aggregated external AUC:", aggregated_external_auc)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Generate a plot of the hyperparameter optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image("optimization_history.png")

# Export the predictions to CSV files
with open(f"{RESULTS_DIR}/{MODEL_DATA_STR}_{FUSION_METHOD}_internal_preds.csv", "w+") as f:
    f.write("model;fusion;dataset;pred;label\n")
    for pred, y in zip(mean_internal_preds, test_y):
        f.write(f"{MODEL_DATA_STR};{FUSION_METHOD};internal;{pred:.4f};{y:.4f}\n")
with open(f"{RESULTS_DIR}/{MODEL_DATA_STR}_{FUSION_METHOD}_external_preds.csv", "w+") as f:
    f.write("model;fusion;dataset;pred;label\n")
    for pred, y in zip(mean_external_preds, external_y):
        f.write(f"{MODEL_DATA_STR};{FUSION_METHOD};external;{pred:.4f};{y:.4f}\n")
