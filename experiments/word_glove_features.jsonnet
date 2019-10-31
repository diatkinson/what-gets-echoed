local word_glove = import "word_glove.jsonnet";
local features_dim = 66;
local glove_dim = word_glove.model.encoder.input_size;

word_glove + {
  "model" +: {
    "use_features": true,
    "encoder" +: {
      "input_size": glove_dim + features_dim,
    }
  },
  "trainer" +: {
    "cuda_device": 2
  }
}