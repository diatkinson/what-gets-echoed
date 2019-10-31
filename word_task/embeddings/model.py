from typing import Dict

import torch
from torch import Tensor

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.training.metrics.f1_measure import F1Measure


@Model.register("word_embeddings_model")
class WordEmbeddingsModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 pos_weight: float,
                 use_features: bool,
                 use_features_post_encoding: bool = False) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.use_features = use_features
        self.use_features_post_encoding = use_features_post_encoding
        self.pos_weight = pos_weight
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(1)

        linear_in_features = encoder.get_output_dim() + (66 if use_features_post_encoding else 0)
        self.transfer_predictor = torch.nn.Linear(in_features=linear_in_features, out_features=2)

    def forward(self, tokens: Dict[str, Tensor],
                features: Tensor,
                is_transferred: Tensor) -> Dict[str, Tensor]:

        mask = get_text_field_mask(tokens)
        embeddings = self.text_field_embedder(tokens)

        if not self.use_features or self.use_features_post_encoding:
            encoded_input = self.encoder(embeddings, mask)
        else:
            encoded_input = self.encoder(torch.cat([embeddings, features], dim=-1), mask)

        if self.use_features_post_encoding:
            tag_logits = self.transfer_predictor(torch.cat([encoded_input, features], dim=-1))
        else:
            tag_logits = self.transfer_predictor(encoded_input)

        output = {"tag_logits": tag_logits}

        if is_transferred is not None:
            self.accuracy(tag_logits, is_transferred, mask)
            self.f1(tag_logits, is_transferred, mask)
            # overweight positive instances
            weight_mask = self.generate_weight_mask(is_transferred, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits,
                                                                is_transferred,
                                                                weight_mask)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self.f1.get_metric(reset)
        return {"acc": self.accuracy.get_metric(reset),
                "prec": precision,
                "recl": recall,
                "f1": f1}

    def generate_weight_mask(self, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hacky. the idea is that if x is self.pos_weight, we want:
        #   (0 + p)m = (1-x)
        #   (1 + p)m = x
        m = 2 * self.pos_weight - 1
        p = (1 - self.pos_weight) / m
        return (gold.float() + p) * m * mask.float()
