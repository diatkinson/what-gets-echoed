import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from tqdm import tqdm
from xgboost import XGBClassifier

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

import extra_scripts.create_vectors as cv
import extra_scripts.utils as utils

TRAIN_FILE = "data/cmv_triples_train_token.jsonlist.gz"
VALID_FILE = "data/cmv_triples_valid_token.jsonlist.gz"
TEST_FILE = "data/cmv_triples_test_token.jsonlist.gz"
GROUP_LABELS = ["group1", "group2_op", "group2_pc", "group3", "group4"]

                # Group 1
FEAT_GROUPS = ([["IDF", "STEM_CHARS", "WORDNET_DEPTH_MIN", "WORDNET_DEPTH_MAX", "TRANSFER_PROB"],
                # Group 2: OP
                ["OP_" + pos for pos in cv.ALL_POS_] +
                ["OP_SUBJ", "OP_OBJ", "OP_OTHER", "OP_IS_ENT", "OP_TF", "OP_NORM_TF", "OP_SFS",
                 "OP_LOC", "OP_QUOTE"],
                # Group 2: PC
                ["PC_" + pos for pos in cv.ALL_POS_] +
                ["PC_SUBJ", "PC_OBJ", "PC_OTHER", "PC_IS_ENT", "PC_TF", "PC_NORM_TF", "PC_SFS",
                 "PC_LOC", "PC_QUOTE"],
                # Group 3
                ["OCCURS_IN_OP_PC", "STEM_POS_DIFF", "STEM_DEP_DIFF", "SFS_IN_OP_NOT_PC",
                 "SFS_IN_PC_NOT_OP"],
                # Group 4
                ["OP_PC_LEN_DIFF", "AVG_TOK_LEN_DIFF", "OP_LEN", "PC_LEN", "OP_PC_POS_DIFF",
                 "DEPTH"]])
LOG_FILE = None


def log(msg):
    log_str_time = f"{datetime.now()} - {msg}"
    print(log_str_time)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


class RandomClassifier:
    def __init__(self, prior=0.5):
        self.prior = prior
        self.best_params_ = prior

    def predict(self, X):
        return np.random.random(len(X)) < self.prior

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.prior = np.mean(y * sample_weight)
        self.best_params_ = self.prior

    def get_params(self):
        return {"prior": self.prior}

    def set_params(self, **params):
        self.prior = params['prior']


def calc_stop_mask(items):
    return np.array([utils.is_stop(item[0]) for item in items])


def calc_doc_mask(doc_type, items):
    # oponly pconly pc op both
    mask = []
    for (_, (_, in_op, in_pc), _, _) in items:
        if in_op and in_pc:
            mask.append({"oponly": 0, "pconly": 0, "pc": 1, "op": 1, "both": 1}[doc_type])
        elif in_op and not in_pc:
            mask.append({"oponly": 1, "pconly": 0, "pc": 0, "op": 1, "both": 0}[doc_type])
        elif not in_op and in_pc:
            mask.append({"oponly": 0, "pconly": 1, "pc": 1, "op": 0, "both": 0}[doc_type])
        else:
            raise Exception
    return np.array(mask)


def calc_pos_weights(items):
    weights = defaultdict(list)
    op_pos_feat_idxs = [cv.FEATURE_LABELS.index("OP_" + pos) for pos in cv.ALL_POS_]
    pc_pos_feat_idxs = [cv.FEATURE_LABELS.index("PC_" + pos) for pos in cv.ALL_POS_]

    for item in tqdm(items):
        features = item[3]
        for pos_idx, pos in enumerate(cv.ALL_POS_):
            weight = (features[op_pos_feat_idxs[pos_idx]] +
                      features[pc_pos_feat_idxs[pos_idx]]) / 2
            weights[pos].append(weight)
    return weights


def getXy(items):
    X = np.array([features for (_, _, _, features) in items])
    y = np.array([transf for (_, (transf, _, _), _, _) in items])
    exp_len_idx = cv.FEATURE_LABELS.index("EXP_LEN")
    # We ignore the explanation length feature, obviously.
    return np.delete(X, exp_len_idx, axis=1), y


def exp_length_performance(clf, X, y, items, mask=None):
    if mask is None:
        mask = np.ones(len(y))

    perf_by_length = {}
    exp_lens = np.array([exp_len for (_, _, exp_len, _) in items])

    # First column is exp_len, second is y, third is stopword_status, rest are features
    combined_arr = np.concatenate([np.stack([exp_lens, y, mask], axis=1), X], axis=1)

    # sort by explanation length
    combined_arr = combined_arr[combined_arr[:, 0].argsort()]
    # split by explanation length
    split_arrs = np.split(combined_arr, np.where(np.diff(combined_arr[:, 0]))[0]+1)

    for arr in tqdm(split_arrs):
        exp_len = arr[0, 0]
        y_len = arr[:, 1]
        mask = arr[:, 2]
        X_len = np.delete(arr, [0, 1, 2], axis=1)
        # store number of stems we predicted on, and performance
        perf_by_length[exp_len] = (len(y_len), score(clf, X_len, y_len, mask))
    return perf_by_length


def score(clf, X, y, mask=None):
    y_pred = clf.predict(X)
    return (f1_score(y, y_pred, sample_weight=mask),
            precision_score(y, y_pred, sample_weight=mask),
            recall_score(y, y_pred, sample_weight=mask))


def log_pos_scores(clf, X, y, pos_weights, stop_mask, doc_mask=None):
    if doc_mask is None:
        doc_mask = np.ones(len(y))
    y_pred = clf.predict(X)
    for word_type, stop_mask in [("all", np.ones(len(y))),
                                 ("stop", stop_mask),
                                 ("content", np.logical_not(stop_mask))]:
        log(f"POS results ({word_type}):")
        for pos in cv.ALL_POS_:
            sample_mask = pos_weights[pos] * doc_mask * stop_mask
            scores = (f1_score(y, y_pred, sample_weight=sample_mask),
                      precision_score(y, y_pred, sample_weight=sample_mask),
                      recall_score(y, y_pred, sample_weight=sample_mask))
            log(f"- {pos} ({int(round(sum(sample_mask)))}): " + f"{scores}")


def fit_or_load(params, X, y, submodel_name, mask=None):
    model_name = f"word_{params['predictor_type']}/model_{submodel_name}.joblib"
    model_fname = "records/" + model_name
    if params['load']:
        try:
            clf = joblib.load(model_fname)
            log(f"Loaded {model_name}.")
            return clf
        except:
            log(f"Couldn't load {model_name}.")

    log(f"Fitting {model_name}.")
    if params['predictor_type'] == "lr":
        lr = LogisticRegression(random_state=0, solver="lbfgs", max_iter=10_000)
        clf = GridSearchCV(lr, params['param_grid'],
                           cv=params['ps_cv'], n_jobs=18, scoring='f1', iid=False, verbose=2)
    elif params['predictor_type'] == 'xgb':
        xgb = XGBClassifier(seed=0, learning_rate=0.1, objective='binary:logistic', gpu_id=2,
                            subsample=1, tree_method="gpu_hist", colsample_bytree=0.8,
                            n_estimators=1000, verbosity=1, nthread=1)
        clf = GridSearchCV(xgb, params['param_grid'], cv=params['ps_cv'],
                           scoring='f1', iid=False, verbose=2, n_jobs=4)
    elif params['predictor_type'] == 'random':
        clf = RandomClassifier()

    clf.fit(X, y, sample_weight=mask)
    log(f"Best params: {clf.best_params_}.")
    log(f"Saving {model_name}.")
    joblib.dump(clf, model_fname)
    return clf


def predict(**fit_params):
    global LOG_FILE
    predictor_type = fit_params["predictor_type"]
    LOG_FILE = f"records/word_{predictor_type}/log.log"

    log(f"####### NEW RUN ({predictor_type}) #######\n\n")

    log("Loading data.")
    train_items = cv.load_prediction_items(TRAIN_FILE)
    valid_items = cv.load_prediction_items(VALID_FILE)
    test_items = cv.load_prediction_items(TEST_FILE)

    log("Extracting raw vectors for sklearn.")
    X_train, y_train = getXy(train_items)
    X_valid, y_valid = getXy(valid_items)
    X_test, y_test = getXy(test_items)

    log("Calculating stop/document masks.")
    stop_mask_train = calc_stop_mask(train_items)
    stop_mask_valid = calc_stop_mask(valid_items)
    stop_mask_test = calc_stop_mask(test_items)
    stop_mask_cv = np.concatenate([stop_mask_train, stop_mask_valid])
    op_only_mask_test = calc_doc_mask("oponly", test_items)
    pc_only_mask_test = calc_doc_mask("pconly", test_items)
    both_mask_test = calc_doc_mask("both", test_items)

    print(f"Test size: {len(test_items)}, stopwords: {sum(stop_mask_test)}, content: {sum(np.logical_not(stop_mask_test))}")
    print(f"Pos: {sum(y_test)}, Neg: {len(y_test) - sum(y_test)}")
    print(f"Stop Pos: {sum(y_test * stop_mask_test)}, Stop Neg: {sum(np.logical_not(y_test) * stop_mask_test)}")
    print(f"Start Pos: {sum(y_test * np.logical_not(stop_mask_test))}, Stop Neg: {sum(np.logical_not(y_test) * np.logical_not(stop_mask_test))}")

    X_cv = np.concatenate([X_train, X_valid], axis=0)
    y_cv = np.concatenate([y_train, y_valid], axis=0)
    # we want 0 for samples in the validation set, -1 otherwise
    valid_markers = np.concatenate([np.zeros(len(X_train)) - 1,
                                    np.zeros(len(X_valid))], axis=0)
    ps_cv = PredefinedSplit(test_fold=valid_markers)
    fit_params['ps_cv'] = ps_cv

    clf = fit_or_load(fit_params, X_cv, y_cv, "all")

    log("Training performance (all, all): " + str(score(clf, X_train, y_train)))
    log("Validation performance (all, all): " + str(score(clf, X_valid, y_valid)))
    log("Testing performance (all, all): " + str(score(clf, X_test, y_test)))

    log("Training performance (all, stop): " +
        str(score(clf, X_train, y_train, stop_mask_train)))
    log("Validation performance (all, stop): " +
        str(score(clf, X_valid, y_valid, stop_mask_valid)))
    log("Testing performance (all, stop): " +
        str(score(clf, X_test, y_test, stop_mask_test)))

    log("Training performance (all, content): " +
        str(score(clf, X_train, y_train, np.logical_not(stop_mask_train))))
    log("Validation performance (all, content): " +
        str(score(clf, X_valid, y_valid, np.logical_not(stop_mask_valid))))
    log("Testing performance (all, content): " +
        str(score(clf, X_test, y_test, np.logical_not(stop_mask_test))))

    log("Testing performance (op, all): " +
        str(score(clf, X_test, y_test, op_only_mask_test)))
    log("Testing performance (op, stop): " +
        str(score(clf, X_test, y_test, op_only_mask_test * stop_mask_test)))
    log("Testing performance (op, content): " +
        str(score(clf, X_test, y_test, op_only_mask_test * np.logical_not(stop_mask_test))))

    log("Testing performance (pc, all): " +
        str(score(clf, X_test, y_test, pc_only_mask_test)))
    log("Testing performance (pc, stop): " +
        str(score(clf, X_test, y_test, pc_only_mask_test * stop_mask_test)))
    log("Testing performance (pc, content): " +
        str(score(clf, X_test, y_test, pc_only_mask_test * np.logical_not(stop_mask_test))))

    log("Testing performance (both, all): " +
        str(score(clf, X_test, y_test, both_mask_test)))
    log("Testing performance (both, stop): " +
        str(score(clf, X_test, y_test, both_mask_test * stop_mask_test)))
    log("Testing performance (both, content): " +
        str(score(clf, X_test, y_test, both_mask_test * np.logical_not(stop_mask_test))))

    log("Calculating POS weights for each sample. (Testing)")
    pos_weights_test = calc_pos_weights(test_items)

    log("POS results for all.")
    log_pos_scores(clf, X_test, y_test, pos_weights_test, stop_mask_test)
    log("POS results for OP.")
    log_pos_scores(clf, X_test, y_test, pos_weights_test, stop_mask_test, op_only_mask_test)
    log("POS results for PC.")
    log_pos_scores(clf, X_test, y_test, pos_weights_test, stop_mask_test, pc_only_mask_test)

    log("Performance by explanation length.")
    f_prefix = f"records/word_{predictor_type}/exp_length_performance_"
    with open(f_prefix + "train.json", "w") as f:
        records = exp_length_performance(clf, X_train, y_train, train_items)
        json.dump(records, f)
    with open(f_prefix + "valid.json", "w") as f:
        records = exp_length_performance(clf, X_valid, y_valid, valid_items)
        json.dump(records, f)
    with open(f_prefix + "test_all.json", "w") as f:
        records = exp_length_performance(clf, X_test, y_test, test_items)
        json.dump(records, f)
    with open(f_prefix + "test_stop.json", "w") as f:
        records = exp_length_performance(clf, X_test, y_test, test_items, stop_mask_test)
        json.dump(records, f)
    with open(f_prefix + "test_content.json", "w") as f:
        records = exp_length_performance(clf, X_test, y_test, test_items,
                                         np.logical_not(stop_mask_test))
        json.dump(records, f)

    log("Running ablation tests.")
    for i, feat_group in enumerate(FEAT_GROUPS):
        feat_idxs = [cv.FEATURE_LABELS.index(feat) for feat in feat_group]
        X_cv_feat = X_cv[:, feat_idxs]
        X_train_feat = X_train[:, feat_idxs]
        X_valid_feat = X_valid[:, feat_idxs]
        X_test_feat = X_test[:, feat_idxs]
        clf = fit_or_load(fit_params, X_cv_feat, y_cv, "forward_" + GROUP_LABELS[i] + "_all")
        log(f"- Forward, {GROUP_LABELS[i]} (training): " +
            str(score(clf, X_train_feat, y_train)))
        log(f"- Forward, {GROUP_LABELS[i]} (validation): " +
            str(score(clf, X_valid_feat, y_valid)))
        log(f"- Forward, {GROUP_LABELS[i]} (testing, all): " +
            str(score(clf, X_test_feat, y_test)))
        log(f"- Forward, {GROUP_LABELS[i]} (testing, stop): " +
            str(score(clf, X_test_feat, y_test, mask=stop_mask_test)))
        log(f"- Forward, {GROUP_LABELS[i]} (testing, content): " +
            str(score(clf, X_test_feat, y_test, mask=np.logical_not(stop_mask_test))))

        # BACK
        X_cv_feat = np.delete(X_cv, feat_idxs, axis=1)
        X_train_feat = np.delete(X_train, feat_idxs, axis=1)
        X_valid_feat = np.delete(X_valid, feat_idxs, axis=1)
        X_test_feat = np.delete(X_test, feat_idxs, axis=1)
        clf = fit_or_load(fit_params, X_cv_feat, y_cv, "backward_" + GROUP_LABELS[i] + "_all")
        log(f"- Backward, {GROUP_LABELS[i]} (training): " +
            str(score(clf, X_train_feat, y_train)))
        log(f"- Backward, {GROUP_LABELS[i]} (validation): " +
            str(score(clf, X_valid_feat, y_valid)))
        log(f"- Backward, {GROUP_LABELS[i]} (testing, all): " +
            str(score(clf, X_test_feat, y_test)))
        log(f"- Backward, {GROUP_LABELS[i]} (testing, stop): " +
            str(score(clf, X_test_feat, y_test, mask=stop_mask_test)))
        log(f"- Backward, {GROUP_LABELS[i]} (testing, content): " +
            str(score(clf, X_test_feat, y_test, mask=np.logical_not(stop_mask_test))))
        log("---")

    log("Broken down group 3 forward results.")
    feat_group = FEAT_GROUPS[3]
    feat_idxs = [cv.FEATURE_LABELS.index(feat) for feat in feat_group]
    X_cv_feat = X_cv[:, feat_idxs]
    X_test_feat = X_test[:, feat_idxs]
    print(X_cv_feat.shape, y_cv.shape)

    clf = fit_or_load(fit_params, X_cv_feat, y_cv, f"forward_{GROUP_LABELS[3]}_all")
    log("Testing performance (all, all): " + str(score(clf, X_test_feat, y_test)))
    log("Testing performance (all, stop): " +
        str(score(clf, X_test_feat, y_test, stop_mask_test)))
    log("Testing performance (all, content): " +
        str(score(clf, X_test_feat, y_test, np.logical_not(stop_mask_test))))

    log("Testing performance (op, all): " +
        str(score(clf, X_test_feat, y_test, op_only_mask_test)))
    log("Testing performance (op, stop): " +
        str(score(clf, X_test_feat, y_test, op_only_mask_test * stop_mask_test)))
    log("Testing performance (op, content): " +
        str(score(clf, X_test_feat, y_test, op_only_mask_test * np.logical_not(stop_mask_test))))

    log("Testing performance (pc, all): " +
        str(score(clf, X_test_feat, y_test, pc_only_mask_test)))
    log("Testing performance (pc, stop): " +
        str(score(clf, X_test_feat, y_test, pc_only_mask_test * stop_mask_test)))
    log("Testing performance (pc, content): " +
        str(score(clf, X_test_feat, y_test, pc_only_mask_test * np.logical_not(stop_mask_test))))

    log("Testing performance (both, all): " +
        str(score(clf, X_test_feat, y_test, both_mask_test)))
    log("Testing performance (both, stop): " +
        str(score(clf, X_test_feat, y_test, both_mask_test * stop_mask_test)))
    log("Testing performance (both, content): " +
        str(score(clf, X_test_feat, y_test, both_mask_test * np.logical_not(stop_mask_test))))

    log("POS results for all.")
    log_pos_scores(clf, X_test_feat, y_test, pos_weights_test, stop_mask_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", action="store_true")
    parser.add_argument("--xgb", action="store_true")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    lr_param_grid = {'C': [10**i for i in range(-1, 5)],
                     'class_weight': [{0: zw, 1: 1-zw}
                                      for zw in [0.25, 0.20, 0.15]]}

    xgb_param_grid = {'scale_pos_weight': [4, 5, 3], 'max_depth': [7, 5, 9],
                      'min_child_weight': [7, 5, 3]}

    if args.lr:
        predict(predictor_type="lr", param_grid=lr_param_grid, load=True)
    if args.xgb:
        predict(predictor_type="xgb", param_grid=xgb_param_grid, load=True)
    if args.random:
        predict(predictor_type="random", load=True)
