import json
from typing import Iterator, Optional, Dict, Any
from nptyping import Array

import numpy as np
from tqdm import tqdm
import logging

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields.text_field import TextField
from allennlp.data.fields.sequence_label_field import SequenceLabelField
from allennlp.data.fields.array_field import ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers import Tokenizer, Token
import random

import extra_scripts.create_vectors as cv

TRANSITION_TOKEN = Token(text="@@transition@@", lemma_="@@transition@@")
TRANSITION_VEC = np.zeros((1,66))

@DatasetReader.register("word_embeddings_reader")
class WordEmbeddingsReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: int = None) -> None:
        super().__init__()
        self._max_seq_len = max_seq_len
        self._token_indexers = token_indexers

    def create_instance(self, triple: Dict[str, Any],
                        op_vec: Array[float],
                        pc_vec: Array[float]) -> Instance:
        op_tokens = cv.get_tokens(triple["op_selftext"])
        pc_tokens = cv.get_tokens(triple["deltaed_comment"])
        exp_tokens = cv.get_tokens(triple["explanation"])
        input_tokens = op_tokens + [TRANSITION_TOKEN] + pc_tokens

        features = np.concatenate([op_vec, TRANSITION_VEC, pc_vec])
        assert len(features) == len(input_tokens)

        exp_stems = {tok.lemma_ for tok in exp_tokens}
        input_stems = [tok.lemma_ for tok in input_tokens]
        transferred = [int(stem in exp_stems) for stem in input_stems]
        assert len(transferred) == len(features)

        text_field = TextField(input_tokens, token_indexers=self._token_indexers)

        return Instance({"tokens": text_field,
                         "features": ArrayField(features),
                         "is_transferred": SequenceLabelField(transferred, text_field)})

    def _read(self, file_path: str) -> Iterator[Instance]:
        tokens = cv.load_gz_file(file_path)
        op_vecs, pc_vecs = cv.load_vectors(file_path + ".npy")
        assert len(op_vecs) == len(pc_vecs) == len(tokens)
        for triple, op_vec, pc_vec in zip(tokens, op_vecs, pc_vecs):
            # We ignore the explanation length feature, obviously.
            exp_len_idx = cv.FEATURE_LABELS.index("EXP_LEN")
            yield self.create_instance(triple,
                                       np.delete(op_vec, exp_len_idx, axis=1),
                                       np.delete(pc_vec, exp_len_idx, axis=1))
