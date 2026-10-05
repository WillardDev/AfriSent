# AfriSenti — Swahili Sentiment Analysis

End-to-end sentiment classification on the [AfriSenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023) Swahili benchmark, covering the full path from raw tweets to trained recurrent classifiers, evaluation, and live inference.

Three classes: `negative`, `neutral`, `positive`.

## Results

Scored on the official held-out test split (748 tweets) by macro-F1.

| Model | Accuracy | Macro-F1 | Negative F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.539 | **0.461** | 0.292 |
| BiLSTM | 0.553 | 0.427 | 0.202 |
| LSTM | 0.455 | 0.413 | 0.284 |
| SimpleRNN | 0.479 | 0.397 | 0.222 |
| GRU | 0.444 | 0.395 | 0.247 |

The sparse n-gram baseline wins. With 1,801 training tweets and a 59/30/11 class imbalance, a randomly initialised recurrent stack has to learn word representations from scratch and overfits before it gets there. The recurrent models are kept as a working, comparable pipeline rather than the recommended production model.

## The main obstacle: class imbalance

| Class | Train tweets | Share | Loss weight |
|---|---|---|---|
| `neutral` | 1,072 | 59.2% | 0.56 |
| `positive` | 547 | 30.2% | 1.10 |
| `negative` | 191 | 10.6% | 3.16 |

A 5.6x gap, handled three ways:

- **Stratified splits** so the 11% `negative` class is present in every slice
- **Inverse-frequency class weights** inside the loss, making a `negative` miss ~5.6x more expensive than a `neutral` one
- **Macro-F1 for model selection** rather than accuracy — always predicting `neutral` already scores 0.59 accuracy, so accuracy is not usable here

Class weights help but do not close the gap: per-class recall on the best model is `neutral` 0.61, `positive` 0.46, `negative` 0.34. The imbalance is still the clearest signal in the per-class breakdown.

## Notebook structure

`AfriSent.ipynb` runs top to bottom with no manual steps.

1. **Environment Setup & Configuration** — imports, seeds, MPS/CUDA auto-detection
2. **Exploratory Data Analysis & Loading** — class distribution, sequence lengths, leakage check
3. **Text Preprocessing & Tokenization** — cleaning, vocabulary from the train split only, padding to `max_length`
4. **Dataset & Data Loader Construction** — stratified 87.5/12.5 split, `DataLoader`s, class weights
5. **Model Architecture Definitions** — one template, four recurrent cells, masked mean pooling
6. **Training Pipeline & Evaluation Hooks** — weighted cross-entropy, AdamW, gradient clipping, `train_model` / `evaluate`, checkpointing on validation macro-F1
7. **Model Execution & Inference** — four architectures plus the baseline, loss and metric plots, confusion matrix, custom-string inference

Each section closes with a short bullet list of what the step produced, including where it fell short.

## Pipeline details

- **Data:** official Swahili `train` / `dev` / `test` TSVs (1,810 / 453 / 748). The dev split is held back as the untouched test set.
- **Preprocessing:** lowercase, strip URLs, `@mentions` and punctuation. Vocabulary is 9,098 words built from train only; `max_length=35` covers 95.6% of tweets; OOV rate 2.0%.
- **Model:** 64-dim embedding → single recurrent layer (32 units) → masked mean pool → linear over 3 classes. Roughly 600k parameters each.
- **Training:** AdamW (lr 2e-3, weight decay 1e-2), batch size 64, early stopping on validation macro-F1 with patience 5 and best-epoch weights restored.
- **Leakage:** 0 duplicate tweets between train and test; 3 between train and dev, which are deduped during cleaning.

## Setup

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn jupyter
```

Then open `AfriSent.ipynb` and run all cells. The dataset downloads from GitHub on first run, so no local data files are needed.

Requires PyTorch — TensorFlow is not used.

## Next steps

- Pretrained or multilingual embeddings, which is the most likely way the recurrent models overtake the baseline
- Focal loss as an alternative to class weighting for the `negative` class
- Per-class threshold tuning to reduce `negative` false positives, the class most prone to false alarms at 3.16x weight
- Code-switched tweets (Swahili mixed with English or Sheng) tokenise poorly against a Swahili-only vocabulary and are a concentrated error source