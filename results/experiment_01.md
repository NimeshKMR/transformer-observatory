# Experiment 1: Character-Level vs BPE Tokenization

## Objective
Measure the effect of BPE tokenization on sequence length and compare the validation cross-entropy of character-level and BPE language models.

## Dataset
- Total characters: 1,115,394
- Training characters: 1,003,854
- Validation characters: 111,540
- Split: chronological 90/10

## Results

| Metric | Character-level | BPE |
|---|---:|---:|
| Vocabulary size | 65 | 565 |
| Training tokens | 1,003,854 | 440,586 |
| Validation tokens | 111,540 | 50,475 |
| Validation loss (nats/token) | 1.7555 | 3.8542 |

BPE represents the training corpus using 56.11% fewer tokens, with an average of 2.278 characters per token.

## Interpretation
BPE substantially reduces the number of token positions required to represent this corpus. This can reduce the effective sequence length for a fixed text passage, although actual runtime and memory savings depend on the model and implementation.

The validation losses are not directly comparable because the models predict different token units and have different vocabulary sizes. A text-level evaluation metric is needed for a fairer comparison.

## Limitations
- Validation loss was estimated using 100 randomly sampled batches.
- The results describe this dataset and these trained checkpoints; they do not establish that BPE is universally superior.
- The reported compression ratio is corpus-specific.
