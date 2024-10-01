from typing import List

import sys
import json

import optuna
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


RANDOM_STATE = 0
FUSION_METHOD = sys.argv[1]  # early or late
MODEL_DATA_STR = sys.argv[2]  # specify model name

TARGET_COLNAME = "cs_pca"  # name of the column containing target labels

BEST_PARAMS = {"all": {"classifier_type": "logistic", "lr_class_weight": "balanced"}}

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

    internal_data = remove_missing(internal_data, ["psa_density", "volume", "psa"])
    external_data = remove_missing(external_data, ["psa_density", "volume", "psa"])

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

    # Don't do any  for the Jackknife analysis
    # Instead, analyze the variance using the optimized parameters from "experiment.py"
    best_params = BEST_PARAMS[feature_set_name].copy()
    print("Using provided parameters:", best_params)

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

    classifier_type = best_params["classifier_type"]
    del best_params["classifier_type"]
    cls = Classifier(classifier_type, seed=RANDOM_STATE)

    cls.fixed_params = {**cls.fixed_params, **best_params}
    cls.train(development_x, development_y)

    internal_test_preds = cls.predict(test_x)
    internal_test_auc = roc_auc_score(test_y, internal_test_preds)

    print("### Jackknifing internal AUC performance ###")
    for colname in development_x.columns:
        print("Variable:", colname)

        # Remove `colname` from the dev and test datasets to measure its importance
        dev_x_left_out = development_x.loc[:, development_x.columns != colname].copy()
        test_x_left_out = test_x.loc[:, development_x.columns != colname].copy()

        jknf_cls = Classifier(classifier_type, seed=RANDOM_STATE)

        # Retrain the model without the variable 
        jknf_cls.fixed_params = {**jknf_cls.fixed_params, **best_params}
        jknf_cls.train(dev_x_left_out, development_y)

        # Obtain the AUC
        jknf_internal_test_preds = jknf_cls.predict(test_x_left_out)
        jknf_internal_test_auc = roc_auc_score(test_y, jknf_internal_test_preds)

        print("AUC without variable:", jknf_internal_test_auc)

        # Create a dataset with ONLY `colname`
        dev_x_left_out = development_x.loc[:, development_x.columns == colname].copy()
        test_x_left_out = test_x.loc[:, development_x.columns == colname].copy()

        jknf_cls = Classifier(classifier_type, seed=RANDOM_STATE)

        # Retrain the model with ONLY this variable
        jknf_cls.fixed_params = {**jknf_cls.fixed_params, **best_params}
        jknf_cls.train(dev_x_left_out, development_y)

        jknf_internal_test_preds = jknf_cls.predict(test_x_left_out)
        jknf_internal_test_auc = roc_auc_score(test_y, jknf_internal_test_preds)

        print("AUC with only variable:", jknf_internal_test_auc)

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

    print("### Jackknifing external AUC performance ###")

    for colname in internal_x.columns:
        print("Variable:", colname)
        dev_x_left_out = internal_x.loc[:, internal_x.columns != colname].copy()
        external_x_left_out = external_x.loc[:, internal_x.columns != colname].copy()

        jknf_cls = Classifier(classifier_type, seed=RANDOM_STATE)

        jknf_cls.fixed_params = {**jknf_cls.fixed_params, **best_params}
        jknf_cls.train(dev_x_left_out, internal_y)

        jknf_internal_test_preds = jknf_cls.predict(external_x_left_out)
        jknf_internal_test_auc = roc_auc_score(external_y, jknf_internal_test_preds)

        print("AUC without variable:", jknf_internal_test_auc)

        dev_x_left_out = internal_x.loc[:, internal_x.columns == colname].copy()
        external_x_left_out = external_x.loc[:, internal_x.columns == colname].copy()

        jknf_cls = Classifier(classifier_type, seed=RANDOM_STATE)

        jknf_cls.fixed_params = {**jknf_cls.fixed_params, **best_params}
        jknf_cls.train(dev_x_left_out, internal_y)

        jknf_internal_test_preds = jknf_cls.predict(external_x_left_out)
        jknf_internal_test_auc = roc_auc_score(external_y, jknf_internal_test_preds)

        print("AUC with only variable:", jknf_internal_test_auc)

print("Done.")