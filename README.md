This repo has the code and data for the 2019 EMNLP paper "[What Gets Echoed? Understanding the 'Pointers' in Explanations of Persuasive Arguments](https://chenhaot.com/papers/explanation-pointers.html)."

To run:
1. Get the data files from the [latest release](https://github.com/diatkinson/what-gets-echoed/releases/tag/v1.0) and place them in the `data` directory.
2. Activate the `cmv-gen` conda environment. (A copy of the environment is in `records/cmv-gen.yml`).
3. Run `python extra_scripts/create_vectors.py` to calculate and store the feature vectors.
4. Find the instructions below for the figure you want to generate.

Feel free to email me at `davatk@gmail.com` if there's something you can't get to work.

# Generating Figures
## Word Transfer Prediction (figs 2a, 2b, 2c, table 4, and table 5)
Everything below is done in the `cmv-gen` conda environment in the project root directory.

The random, logistic regression, and xgb results can be gotten with `python word_task/non_neural/predict.py --random` (or `--lr`, or `--xgb`). Models are loaded from `records/word_random/*.joblib` (or `word_lr`, or `word_xgb`), or, if those files aren't there, they'll be created and placed there. They store their results in the same directories, in the `log.log` file, which is ultimately where Tables 4 and 5 come from. New results are appended, so make sure you're looking at the right section of the file.

To get results for just the `IN_OP_AND_PC` feature, run `python word_task/non_neural/predict_single_feature.py`. This one just sends its results to stdout.

To train the vanilla LSTM model, run `allennlp train experiments/word_glove.jsonnet -s records/word_glove --include-package word_task` (make sure `records/word_glove` is empty). You can change hyperparameters in `experiments/word_glove.jsonnet`. Records will be stored in `records/word_task/stdout.log`.

To evaluate (on the test set, in this case), run `python word_task/embeddings/evaluate.py records/word_glove data/cmv_triples_test_token.jsonlist.gz --gpu 2`. Records are stored in `records/word_task/evaluate.log`.

To train and evaluate the LSTM+features model, just do the above, but replace `word_glove` with `word_glove_features`.

The actual plots are done in notebooks:
* Figure 2a (overall performance): `Paper - Overall Performance.ipynb`
* Figure 2b (group 3 feature importance): `Paper - XBG Feature Importance.ipynb`
* Figure 2c (performance by word source): `Paper - OP, PC performance.ipynb`
## Explanation Generation (table 6)
Coming later!
## Everything Else (figs 1a, 1b, 1c, table 3)
These are in notebooks.

* Figure 1a (length correlations): `Paper - Length correlations (heatmap).ipynb`
* Figure 1b (echo sources): `Paper - Source of Explanation Token.ipynb`
* Figure 1c (echo prob. vs. freq): `Paper - Echoing Probability vs. Document Frequency.ipynb`
* Table 3: `Paper - Feature Significance.ipynb`
