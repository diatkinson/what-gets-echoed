from predict import *
import predict
import extra_scripts.create_vectors as cv

predict.LOG_FILE = "../records/word_xgb/single_feature.log"

log("##### NEW RUN #####\n")
train_items = cv.load_prediction_items(TRAIN_FILE)
valid_items = cv.load_prediction_items(VALID_FILE)
test_items = cv.load_prediction_items(TEST_FILE)

X_train, y_train = getXy(train_items)
X_valid, y_valid = getXy(valid_items)
X_test, y_test = getXy(test_items)

stop_mask_train = calc_stop_mask(train_items)
stop_mask_valid = calc_stop_mask(valid_items)
stop_mask_test = calc_stop_mask(test_items)
stop_mask_cv = np.concatenate([stop_mask_train, stop_mask_valid])
op_only_mask_test = calc_doc_mask("oponly", test_items)
pc_only_mask_test = calc_doc_mask("pconly", test_items)
both_mask_test = calc_doc_mask("both", test_items)

X_cv = np.concatenate([X_train, X_valid], axis=0)
y_cv = np.concatenate([y_train, y_valid], axis=0)
# we want 0 for samples in the validation set, -1 otherwise
valid_markers = np.concatenate([np.zeros(len(X_train)) - 1,
                                np.zeros(len(X_valid))], axis=0)
ps_cv = PredefinedSplit(test_fold=valid_markers)

xgb_param_grid = {'scale_pos_weight': [4, 5, 3], 'max_depth': [7, 5, 9],
                  'min_child_weight': [7, 5, 3]}

fit_params = {"predictor_type": "xgb", "param_grid": xgb_param_grid,
              'load': True, 'ps_cv': ps_cv}
feat = "OCCURS_IN_OP_PC"
feat_idx = [cv.FEATURE_LABELS.index(feat)]
X_cv_feat = X_cv[:, feat_idx]
X_train_feat = X_train[:, feat_idx]
X_valid_feat = X_valid[:, feat_idx]
X_test_feat = X_test[:, feat_idx]
print(X_cv_feat.shape, y_cv.shape)

clf = fit_or_load(fit_params, X_cv_feat, y_cv, f"only_{feat}_all")
print(f"- Forward, {feat} (training): " +
      str(score(clf, X_train_feat, y_train)))
print(f"- Forward, {feat} (validation): " +
      str(score(clf, X_valid_feat, y_valid)))
print(f"- Forward, {feat} (testing, all): " +
      str(score(clf, X_test_feat, y_test)))
print(f"- Forward, {feat} (testing, stop): " +
      str(score(clf, X_test_feat, y_test, mask=stop_mask_test)))
print(f"- Forward, {feat} (testing, content): " +
      str(score(clf, X_test_feat, y_test, mask=np.logical_not(stop_mask_test))))
