local embedding_dim = 100;
local hidden_dim = 300;

{
  "dataset_reader": {
    "type": "word_embeddings_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    },
  },
  "train_data_path": "data/cmv_triples_train_token.jsonlist.gz",
  "validation_data_path": "data/cmv_triples_valid_token.jsonlist.gz",
  "model": {
    "type": "word_embeddings_model",
    "pos_weight": 0.8,
    "use_features": false,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 100,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "trainable": true
      },
    },
    "encoder": {
      "type": "alternating_lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim,
      "num_layers": 1,
      "use_highway": true
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32,
  },
  "trainer": {
    "cuda_device": 2,
    "num_epochs": 2,
    "validation_metric": "+f1",
    "optimizer": "adam",
  },
}
