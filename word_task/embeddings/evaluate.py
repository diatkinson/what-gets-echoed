from typing import DefaultDict, List
from nptyping import Array

import os
import logging
import numpy as np
from collections import defaultdict
import random
import argparse
import re
from tqdm import tqdm

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.special import softmax

from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common import Params

import extra_scripts.utils as utils

GLOVE = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"
EMBEDDING_SOURCES = {"text_field_embedder.token_embedder_tokens": GLOVE}


def decode_logits(stems: List[str], logits: Array) -> List[bool]:
    """
    Input is map(stemmer.stem, document) (i.e., possibly repeated stems)
    Output is the predicted transfer status of each possible stem, sorted by stem
    """
    assert len(stems) == len(logits)
    stem_logits = defaultdict(list)
    # collect and average logits for each possible stem
    for stem, logit in zip(stems, logits):
        # Softmax so that each stem has equal weight
        stem_logits[stem].append(softmax(logit))
    # Predict transfer if the mean of col 1 is higher than col 0
    stem_set = list(sorted(set(stems)))
    for stem in stem_set:
        stem_logits[stem] = np.mean(np.array(stem_logits[stem]), axis=0)
    return [stem_logits[stem][1] > stem_logits[stem][0] for stem in stem_set]


def update_perf_records(perf_records: DefaultDict[str, List[bool]],
                        stems: List[str],
                        pred_transf: List[bool],
                        gold_transf: List[bool]) -> None:
    assert len(stems) == len(pred_transf) == len(gold_transf)
    for stem in set(stems):
        if re.match(r"\w+", stem):
            stem_idx = stems.index(stem)
            gold = gold_transf[stem_idx]
            pred = pred_transf[stem_idx]
            perf_records["gold_all"].append(gold)
            perf_records["pred_all"].append(pred)
            if utils.is_stop(stem):
                perf_records["gold_stop"].append(gold)
                perf_records["pred_stop"].append(pred)
            else:
                perf_records["gold_content"].append(gold)
                perf_records["pred_content"].append(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_dir", type=str)
    parser.add_argument("token_file", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    random.seed(0)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(args.archive_dir, "evaluate.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    archive = load_archive(os.path.join(args.archive_dir, "model.tar.gz"))
    model = archive.model
    dataset_reader = DatasetReader.from_params(archive.config["dataset_reader"])
    logging.info(f"Creating instances from {args.token_file}.")
    instances = dataset_reader.read(args.token_file)
    random.shuffle(instances)

    model.vocab.extend_from_instances(Params({}), instances=instances)
    model.extend_embedder_vocab(EMBEDDING_SOURCES)
    model.cuda(args.gpu)

    logging.info("Evaluating.")
    perf_records: DefaultDict[str, List[bool]] = defaultdict(list)
    with tqdm(total=len(instances)) as pbar:
        for inst in instances:
            stems = [tok.lemma_ for tok in inst["tokens"]]
            logits = model.forward_on_instance(inst)["tag_logits"]
            pred_transf = decode_logits(stems, logits)
            stem_set = list(sorted(set(stems)))
            gold_transf = [inst["is_transferred"][stems.index(stem)] for stem in stem_set]
            assert len(gold_transf) == len(pred_transf)
            update_perf_records(perf_records, stem_set, pred_transf, gold_transf)

            gold_all = perf_records["gold_all"]
            pred_all = perf_records["pred_all"]

            prec = precision_score(gold_all, pred_all)
            recl = recall_score(gold_all, pred_all)
            f1 = f1_score(gold_all, pred_all)

            pbar.set_postfix(f1=f1, prec=prec, recl=recl)
            pbar.update()

    logging.info("All words:")
    gold = perf_records["gold_all"]
    pred = perf_records["pred_all"]
    prec = precision_score(gold, pred)
    recl = recall_score(gold, pred)
    f1 = f1_score(gold, pred)
    acc = accuracy_score(gold, pred)
    logging.info(f"F1: {f1}, precision: {prec}, recall: {recl}, acc: {acc}.")

    logging.info("Just content words:")
    gold = perf_records["gold_content"]
    pred = perf_records["pred_content"]
    prec = precision_score(gold, pred)
    recl = recall_score(gold, pred)
    f1 = f1_score(gold, pred)
    acc = accuracy_score(gold, pred)
    logging.info(f"F1: {f1}, precision: {prec}, recall: {recl}, acc: {acc}.")

    logging.info("Just stopwords:")
    gold = perf_records["gold_stop"]
    pred = perf_records["pred_stop"]
    prec = precision_score(gold, pred)
    recl = recall_score(gold, pred)
    f1 = f1_score(gold, pred)
    acc = accuracy_score(gold, pred)
    logging.info(f"F1: {f1}, precision: {prec}, recall: {recl}, acc: {acc}.")
