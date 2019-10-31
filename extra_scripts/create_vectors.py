'''
Basic usage:
- Load the tokenized data into Allennlp Token (idx is set to -1, donot use).
    - Tokenized data is pipe delimited in following order :
        'text', 'lemma_', 'pos_', 'ent_type_', 'tag_', 'dep_'
- Scale the loaded data in place:
      scale_vectors(train_op, train_delta, valid_op, valid_delta)

- This gives you, e.g., train_op, which is a list (of length: # of training instances) where each
  element is an array of shape (number of tokens in that OP instance, number of features)
'''

import gzip
import json
import logging
import math
import re
import traceback
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from allennlp.data.tokenizers import Token
from nptyping import Array

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)

IGNORE_ENTITIES = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL",
                   "CARDINAL"]
SUBJ_DEP_LABELS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJ_DEP_LABELS = ["dobj", "dative", "attr", "oprd"]

"""
change this if needed, new tokenized data has following keys.
"""
OP_TAG = "op_selftext"
PC_TAG = "deltaed_comment"
EXP_TAG = "explanation"
DEPTH_TAG = "depth"

ALL_POS_ = ['ADP', 'PRON', 'X', 'DET', 'ADJ', 'PROPN', 'VERB', 'PART',
            'CCONJ', 'INTJ', 'NOUN', 'NUM', 'ADV', 'PUNCT', 'SYM', 'AUX']

attrs = ['text', 'lemma_', 'pos_', 'ent_type_', 'tag_', 'dep_']
QUOTE = ["\"".encode('utf-8'), "“".encode('utf-8'), "”".encode('utf-8')]

FEATURE_LABELS = (["OP_PC_LEN_DIFF", "AVG_TOK_LEN_DIFF", "OP_LEN", "PC_LEN", "OP_PC_POS_DIFF",
                   "DEPTH", "IDF", "STEM_CHARS", "WORDNET_DEPTH_MIN", "WORDNET_DEPTH_MAX",
                   "TRANSFER_PROB"] +
                  ["OP_" + pos for pos in ALL_POS_] + ["PC_" + pos for pos in ALL_POS_] +
                  ["OP_SUBJ", "OP_OBJ", "OP_OTHER", "PC_SUBJ", "PC_OBJ", "PC_OTHER", "OP_IS_ENT",
                   "PC_IS_ENT", "OP_TF", "OP_NORM_TF", "PC_TF", "PC_NORM_TF", "OP_SFS", "PC_SFS",
                   "OP_LOC", "PC_LOC", "OP_QUOTE", "PC_QUOTE", "OCCURS_IN_OP_PC", "STEM_POS_DIFF",
                   "STEM_DEP_DIFF", "SFS_IN_OP_NOT_PC", "SFS_IN_PC_NOT_OP", "EXP_LEN"])


def get_tokens(text: List[List[str]], max_len: int = -1) -> List[Token]:
    """
    Input is a list of tokens, each token comprising [text, lemma_, pos_, etc]
    """
    tokens_list = []
    if max_len != -1:
        text = text[:max_len]
    for word_features in text:
        tokens_list.append(Token(word_features[0], -1, word_features[1],
                                 word_features[2], word_features[4],
                                 word_features[5], word_features[3]))
    return tokens_list


def compute_min_wordnet_depth(lemma):
    temp = [curr_syn.min_depth() for curr_syn in wn.synsets(lemma)]
    if len(temp) == 0:
        return 0
    return np.mean(temp)


def compute_max_wordnet_depth(lemma):
    temp = [curr_syn.max_depth() for curr_syn in wn.synsets(lemma)]
    if len(temp) == 0:
        return 0
    return np.mean(temp)


class Vocab:
    def __init__(self, data: List[Dict[str, Any]], max_len=1000):
        pos_pairs = []
        self.num_docs = len(data) * 2
        self.lemma_seen: DefaultDict[str, int] = defaultdict(int)
        self.lemma_transferred: DefaultDict[str, int] = defaultdict(int)
        for inst in tqdm(data):
            op_tokens = get_tokens(inst[OP_TAG], max_len)
            op_lemmas = [tok.lemma_ for tok in op_tokens]
            delta_tokens = get_tokens(inst[PC_TAG], max_len)
            delta_lemmas = [tok.lemma_ for tok in delta_tokens]
            exp_tokens = get_tokens(inst[EXP_TAG], max_len)
            exp_lemmas = [tok.lemma_ for tok in exp_tokens]

            for lemma in op_lemmas + delta_lemmas:
                self.lemma_seen[lemma] += 1
                if lemma in exp_lemmas:
                    self.lemma_transferred[lemma] += 1
            for tok in op_tokens + delta_tokens + exp_tokens:
                if tok.pos_ and tok.tag_:
                    pos_pairs.append((tok.pos_, tok.tag_))

        transfer_probs = [self.lemma_transferred[lemma] / self.lemma_seen[lemma]
                          for lemma in self.lemma_seen]
        self.transfer_prior = np.mean(transfer_probs)

        # We shouldn't have any POS tags not already in ALL_POS_
        self.all_pos = list(ALL_POS_)
        for curr in list(sorted(pos_ for pos_, tag_ in pos_pairs)):
            self.all_pos.append(curr)
        if len(set(self.all_pos)) != len(set(ALL_POS_)):
            print('pos tags : ', self.all_pos)

        self.all_pos = list(self.all_pos)

    def idf(self, lemma: str) -> float:
        try:
            return math.log(self.num_docs / self.lemma_seen[lemma])
        except ZeroDivisionError:
            return 0

    def transfer_prob(self, lemma: str) -> float:
        try:
            return self.lemma_transferred[lemma] / self.lemma_seen[lemma]
        except ZeroDivisionError:
            return self.transfer_prior


def tf(count: int) -> float:
    return 1 + math.log(count) if count > 0 else 0


def js_divergence(x_probs: List[float], y_probs: List[float]) -> float:
    assert (abs(sum(x_probs) - 1) < 0.0001) and (abs(sum(y_probs) - 1) < 0.0001)
    # JS distance is sqrt(JS divergence). scipy gives us distance
    return jensenshannon(x_probs, y_probs) ** 2


def compute_distribution(lst: List[Any]) -> List[float]:
    lst_sum = sum(lst)
    if lst_sum == 0:
        return [1 / len(lst) for _ in lst]
    else:
        return [x / lst_sum for x in lst]


def get_inquote_lemmas(tokens_list, in_quotes_tokens):
    is_in_quote = 0
    for tok in tokens_list:
        if is_in_quote == 1 and tok.text.encode('utf-8') not in QUOTE:
            in_quotes_tokens.append(tok.text)
        if is_in_quote == 1 and tok.text.encode('utf-8') in QUOTE:
            is_in_quote = 0
        elif is_in_quote == 0 and tok.text.encode('utf-8') in QUOTE:
            is_in_quote = 1
    return in_quotes_tokens


def create_token_vectors_inst(inst,
                              vocab: Vocab,
                              max_len=1000) -> Tuple[Array[float], Array[float]]:
    op_tokens = get_tokens(inst[OP_TAG], max_len)
    delta_tokens = get_tokens(inst[PC_TAG], max_len)
    depth = inst[DEPTH_TAG]
    exp_len = len(get_tokens(inst[EXP_TAG]))

    # Precalculate for performance
    op_lemmas = []
    op_lemma_count: Counter = Counter([])
    op_pos_count_all = [0 for _ in range(len(ALL_POS_))]
    op_token_lens = []
    for tok in op_tokens:
        op_lemmas.append(tok.lemma_)
        op_lemma_count[tok.lemma_] += 1
        op_token_lens.append(len(tok.text))
        op_pos_count_all[ALL_POS_.index(tok.pos_)] += 1
    delta_lemmas = []
    delta_lemma_count: Counter = Counter([])
    delta_token_lens = []
    delta_pos_count_all = [0 for _ in range(len(ALL_POS_))]
    for tok in delta_tokens:
        delta_lemmas.append(tok.lemma_)
        delta_lemma_count[tok.lemma_] += 1
        delta_token_lens.append(len(tok.text))
        delta_pos_count_all[ALL_POS_.index(tok.pos_)] += 1

    delta_in_quotes_lemmas = get_inquote_lemmas(delta_tokens, [])
    op_in_quotes_lemmas = get_inquote_lemmas(op_tokens, [])

    assert len(op_lemmas) != 0 and len(delta_lemmas) != 0

    # Group 4: General OP and PC properties (6)
    # -------------------------------------

    general_inst_features: List[float] = []
    # Length difference (1)
    general_inst_features.append(abs(len(op_tokens) - len(delta_tokens)))
    # Average word length difference (1)
    general_inst_features.append(abs(np.mean(op_token_lens) - np.mean(delta_token_lens)))
    # OP Length (1)
    general_inst_features.append(len(op_tokens))
    # PC Length (1)
    general_inst_features.append(len(delta_tokens))
    # POS distributional difference (for all lemmas) (1)
    general_inst_features.append(js_divergence(compute_distribution(op_pos_count_all),
                                               compute_distribution(delta_pos_count_all)))
    # Depth (1)
    general_inst_features.append(depth)


    num_features = 67 # G1: 5 + G2: 50 + G3: 5 + G4: 6 + Exp len: 1
    op_vec = np.zeros((len(op_tokens), num_features))
    delta_vec = np.zeros((len(delta_tokens), num_features))

    for i, lemma in enumerate(op_lemmas + delta_lemmas):
        features = list(general_inst_features)

        # Precalculated for performance
        op_pos: Counter = Counter()
        op_deps: Counter = Counter()
        op_surface_forms = set()
        op_entities = 0
        op_locs = []
        for loc, tok in enumerate(op_tokens):
            if tok.lemma_ == lemma:
                op_pos[tok.pos_] += 1
                op_deps[tok.dep_] += 1
                op_surface_forms.add(tok.text)
                if tok.ent_type_ != "" and tok.ent_type_ not in IGNORE_ENTITIES:
                    op_entities += 1
                op_locs.append(loc / len(op_tokens))
        delta_pos: Counter = Counter()
        delta_deps: Counter = Counter()
        delta_surface_forms = set()
        delta_entities = 0
        delta_locs = []
        for loc, tok in enumerate(delta_tokens):
            if tok.lemma_ == lemma:
                delta_pos[tok.pos_] += 1
                delta_deps[tok.dep_] += 1
                delta_surface_forms.add(tok.text)
                if tok.ent_type_ != "" and tok.ent_type_ not in IGNORE_ENTITIES:
                    delta_entities += 1
                delta_locs.append(loc / len(delta_tokens))


        # Group 1: Lemma level properties (5)
        # -------------------------------

        # IDF (1)
        features.append(vocab.idf(lemma))
        # Number of characters in lemma (1)
        features.append(len(lemma))
        # Wordnet depth (2)
        features.append(compute_min_wordnet_depth(lemma))
        features.append(compute_max_wordnet_depth(lemma))
        # Transfer probability (1)
        features.append(vocab.transfer_prob(lemma))
        assert len(features) == 6 + 5


        # Group 2: How the lemma is used in OP/PC (50)
        # ---------------------------------------

        # OP POS distribution (16)
        op_pos_count = np.array([op_pos[pos_] for pos_ in ALL_POS_])
        op_pos_dist = compute_distribution(op_pos_count)
        features += op_pos_dist
        # Delta POS distribution (16)
        delta_pos_count = np.array([delta_pos[pos_] for pos_ in ALL_POS_])
        delta_pos_dist = compute_distribution(delta_pos_count)
        features += delta_pos_dist
        # OP dependency label distribution (3)
        op_subj, op_obj, op_other = 0, 0, 0
        for dep, dep_count in op_deps.items():
            if dep in SUBJ_DEP_LABELS:
                op_subj += dep_count
            elif dep in OBJ_DEP_LABELS:
                op_obj += dep_count
            else:
                op_other += dep_count
        op_dep_counts = [op_subj, op_obj, op_other]
        op_dep_dist = compute_distribution(op_dep_counts)
        features += op_dep_dist
        # Delta dependency label distribution (3)
        delta_subj, delta_obj, delta_other = 0, 0, 0
        for dep, dep_count in delta_deps.items():
            if dep in SUBJ_DEP_LABELS:
                delta_subj += dep_count
            elif dep in OBJ_DEP_LABELS:
                delta_obj += dep_count
            else:
                delta_other += dep_count
        delta_dep_counts = [delta_subj, delta_obj, delta_other]
        delta_dep_dist = compute_distribution(delta_dep_counts)
        features += delta_dep_dist
        # OP and Delta: entity percentage (2)
        features.append(op_entities / len(op_tokens))
        features.append(delta_entities / len(delta_tokens))
        # OP term frequency and normalized tf (2)
        op_tf = tf(op_lemma_count[lemma])
        features.append(op_tf)
        features.append(op_tf / len(op_tokens))
        # Delta term frequency and normalized tf (2)
        delta_tf = tf(delta_lemma_count[lemma])
        features.append(delta_tf)
        features.append(delta_tf / len(delta_tokens))
        # OP and Delta: number of surface forms (2)
        features.append(len(op_surface_forms))
        features.append(len(delta_surface_forms))
        # OP and Delta: location (2)
        features.append(np.mean(op_locs) if op_locs else 0.5)
        features.append(np.mean(delta_locs) if delta_locs else 0.5)
        # OP and Delta: is in quote (2)
        features.append(op_in_quotes_lemmas.count(lemma))
        features.append(delta_in_quotes_lemmas.count(lemma))
        assert len(features) == 6 + 5 + 50

        # Group 3: How the lemma connects the OP and PC (5)
        # ---------------------------------------------

        # Occurrence: does it appear in both OP and PC (1)
        features.append(int(lemma in op_lemmas and lemma in delta_lemmas))
        # JS divergence between OP/PC POS distributions for that lemma (1)
        features.append(js_divergence(op_pos_dist, delta_pos_dist))
        # JS divergence between OP/PC dependency distributions for that lemma(1)
        features.append(js_divergence(op_dep_dist, delta_dep_dist))
        # Surface form difference (2)
        features.append(len(op_surface_forms - delta_surface_forms))
        features.append(len(delta_surface_forms - op_surface_forms))


        # Store explanation length, for later performance analysis (1)
        features.append(exp_len)
        assert len(features) == 6 + 5 + 50 + 5 + 1 == len(FEATURE_LABELS)


        if i < len(op_tokens):
            op_vec[i] = np.array(features)
        else:
            delta_vec[i - len(op_tokens)] = np.array(features)

    return op_vec, delta_vec


def create_token_vectors(data,
                         vocab: Vocab,
                         max_len=1000) -> Tuple[List[Array[float]], List[Array[float]]]:
    op_vecs = []
    delta_vecs = []
    for inst in tqdm(data):
        try:
            op_vec, delta_vec = create_token_vectors_inst(inst, vocab, max_len)
        except ValueError:
            traceback.print_exc()
        op_vecs.append(op_vec)
        delta_vecs.append(delta_vec)
    return op_vecs, delta_vecs

def load_vectors(fname: str) -> Tuple[List[Array[float]], List[Array[float]]]:
    logging.info(f"Loading {fname}.")
    vectors = np.load(fname)
    logging.info("Splitting vectors into instances.")
    # This splits the vectors by the value of column 1 (the triple index).
    split_vecs = np.split(vectors, np.where(np.diff(vectors[:, 1]))[0]+1)
    ops, deltas = [], []
    for inst_vecs in tqdm(split_vecs):
        # column 0 contains 0 for op and 1 for delta
        op_vecs = inst_vecs[inst_vecs[:, 0] == 0]
        delta_vecs = inst_vecs[inst_vecs[:, 0] == 1]
        # remove columns 0 and 1, to get back to original vectors
        ops.append(np.delete(op_vecs, [0, 1], axis=1))
        deltas.append(np.delete(delta_vecs, [0, 1], axis=1))
    return ops, deltas


def scale_vectors(train_ops: List[Array[float]],
                  train_deltas: List[Array[float]],
                  valid_ops: List[Array[float]],
                  valid_deltas: List[Array[float]],
                  test_ops: List[Array[float]],
                  test_deltas: List[Array[float]]) -> None:
    train_vectors = np.concatenate(train_ops + train_deltas)
    scaler = MinMaxScaler(copy=False)
    scaler.fit(train_vectors)
    for arr in train_ops + train_deltas + valid_ops + valid_deltas + test_ops + test_deltas:
        scaler.transform(arr)


def save_vectors(op: List[Array[float]], delta: List[Array[float]],
                 fname: str) -> None:
    assert len(op) == len(delta)
    logging.info("Inserting location and text type information into vectors.")
    combined_vecs_lst = []
    for inst_idx in range(len(op)):
        # insert column containing the instance number
        op_arr = np.insert(op[inst_idx], 0, inst_idx, axis=1)
        delta_arr = np.insert(delta[inst_idx], 0, inst_idx, axis=1)
        # insert column of 0s or 1s, telling us these are OP (0) or Delta (1) vectors
        op_arr = np.insert(op_arr, 0, 0, axis=1)
        delta_arr = np.insert(delta_arr, 0, 1, axis=1)

        combined_vecs_lst.append(op_arr)
        combined_vecs_lst.append(delta_arr)

    combined_vecs = np.concatenate(combined_vecs_lst)
    logging.info(f"Saving array of shape {combined_vecs.shape} to {fname}.")
    np.save(fname, combined_vecs)


def load_gz_file(filename: str) -> List[Dict[str, Any]]:
    logging.info(f"Loading {filename}.")
    data = []
    with gzip.open(filename, 'r') as ifile:
        for curr in tqdm(ifile.readlines()):
            data.append(json.loads(curr))
    return data


def load_tokenized_data(fname: str, max_len: int = -1) -> List[Dict[str, List[Token]]]:
    json_data = load_gz_file(fname)
    logging.info(f"Tokenizing {fname}.")
    tokenized_data = []
    for inst in tqdm(json_data):
        tokenized_inst = {"depth": inst["depth"]}
        for key in inst:
            if key != 'depth' and key != 'link_id':
                tokenized_inst[key] = get_tokens(inst[key], max_len)
        tokenized_data.append(tokenized_inst)
    return tokenized_data

def load_prediction_items(fname: str, test: bool = False) -> List[Tuple[str, Tuple[bool, bool, bool], Array[float]]]:
    """
    We expect a tokenized .jsonlist.gz file as input.
    We return a list of tuples, each of form (lemma, (stem in Exp, stem in OP, stem in PC), Exp length, numpy array of features)
    """
    tokens = load_gz_file(fname)
    op_vecs, pc_vecs = load_vectors(fname + ".npy")
    test_str = " (and verifying feature vectors are the same between same-stem words)"
    logging.info(f"Extracting unique stems{test_str if test else ''}.")
    pred_items = []
    for i, _ in enumerate(tqdm(tokens)):
        ex_stems = {tok[1] for tok in tokens[i]["explanation"]}
        op_stems = [tok[1] for tok in tokens[i]["op_selftext"]]
        pc_stems = [tok[1] for tok in tokens[i]["deltaed_comment"]]
        for stem in set(op_stems + pc_stems):
            if not re.match(r"\w+", stem):
                continue
            if test:
                op_features = [feat for wd_idx, feat in enumerate(op_vecs[i])
                               if op_stems[wd_idx] == stem]
                pc_features = [feat for wd_idx, feat in enumerate(pc_vecs[i])
                               if pc_stems[wd_idx] == stem]
                all_features = op_features + pc_features
                assert all(np.array_equal(all_features[0], feat) for feat in all_features)
            # In theory, within an instance, all feature vectors for the same stem will be the same.
            # So we just choose an arbitrary one.
            if stem in op_stems:
                features = op_vecs[i][op_stems.index(stem)]
            else:
                features = pc_vecs[i][pc_stems.index(stem)]
            pred_items.append((stem, (stem in ex_stems, stem in op_stems, stem in pc_stems),
                               len(tokens[i]["explanation"]), features))
    return pred_items

def create_and_save_vectors(train_fname: str, valid_fname: str,
                            test_fname: str, max_seq_len=-1):
    train_json = load_gz_file(train_fname)
    valid_json = load_gz_file(valid_fname)
    test_json = load_gz_file(test_fname)
    logging.info("Creating vocabulary.")
    vocab = Vocab(train_json, max_seq_len)
    logging.info("Calculating feature vectors for training data.")
    op_train_vecs, delta_train_vecs = create_token_vectors(train_json, vocab, max_seq_len)
    logging.info("Calculating feature vectors for validation data.")
    op_valid_vecs, delta_valid_vecs = create_token_vectors(valid_json, vocab, max_seq_len)
    logging.info("Calculating feature vectors for testing data.")
    op_test_vecs, delta_test_vecs = create_token_vectors(test_json, vocab, max_seq_len)
    logging.info("Scaling vectors to lie in [0,1].")
    scale_vectors(op_train_vecs, delta_train_vecs, op_valid_vecs, delta_valid_vecs, op_test_vecs,
                  delta_test_vecs)
    logging.info("Saving training vectors.")
    save_vectors(op_train_vecs, delta_train_vecs, train_fname + ".npy")
    logging.info("Saving validation vectors.")
    save_vectors(op_valid_vecs, delta_valid_vecs, valid_fname + ".npy")
    logging.info("Saving testing vectors.")
    save_vectors(op_test_vecs, delta_test_vecs, test_fname + ".npy")

    return (op_train_vecs, delta_train_vecs, op_valid_vecs, delta_valid_vecs, op_test_vecs,
            delta_test_vecs)


if __name__ == '__main__':
    TRAIN_DATA_PATH = "data/cmv_triples_train_token.jsonlist.gz"
    VALID_DATA_PATH = "data/cmv_triples_valid_token.jsonlist.gz"
    TEST_DATA_PATH = "data/cmv_triples_test_token.jsonlist.gz"
    create_and_save_vectors(TRAIN_DATA_PATH, VALID_DATA_PATH, TEST_DATA_PATH)
