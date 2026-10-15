# Transformer Observatory

### From implementing a Transformer from scratch to inspecting its internal computations.

Transformer Observatory is an interactive inspection workbench for a small GPT-style decoder-only Transformer. It combines a from-scratch NumPy implementation, a trained PyTorch language model, internal activation instrumentation, and an interactive interface for exploring attention, hidden states, token predictions, and model diagnostics.

The goal was not simply to train a language model, but to understand how its components work and build tools to investigate its behavior.

---

## Overview

Transformer Observatory allows you to inspect a trained language model at multiple levels:

- **Attention:** Visualize attention matrices across layers and heads.
- **Q, K and V:** Inspect query, key and value representations.
- **Hidden states:** Explore intermediate representations throughout the network.
- **Next-token predictions:** Examine logits and probability distributions.
- **Diagnostics:** Investigate activation statistics and representation norms.
- **Experiments:** Compare tokenization strategies and quantify attention behavior.

The project follows a progression from mathematical understanding to implementation, training, instrumentation, visualization, and experimentation.

## Features

### Interactive Transformer inspection

Explore the model through an interactive Streamlit interface.

- Select a prompt and inspect its tokenization.
- Choose a Transformer layer and attention head.
- Inspect attention weights and individual query-token attention patterns.
- Examine Q/K/V vectors and hidden-state representations.
- View next-token probability distributions and ranked predictions.
- Investigate activation and norm diagnostics.

### From-scratch implementation

Core Transformer components were first implemented using NumPy to understand the underlying computations before moving to PyTorch.

Components include:

- Tokenization and embeddings
- Scaled dot-product self-attention
- Multi-head attention
- Causal masking and softmax
- Layer normalization
- Feed-forward MLP with GELU
- Residual connections and pre-LayerNorm blocks
- Language-model output projection
- Cross-entropy loss

### Trained GPT-style language model

A small decoder-only GPT was implemented in PyTorch and trained on the Tiny Shakespeare corpus.

The project includes character-level and BPE tokenization experiments, trained checkpoints, and utilities for generating text and inspecting model internals.

### Model instrumentation

The Observatory captures intermediate computations during inference, including attention weights, Q/K/V tensors, hidden states, logits, and next-token probabilities.

This makes the model inspectable beyond its final generated output.

---

## Architecture

```text
                    Input prompt
                         |
                         v
                      Tokenizer
                         |
                         v
               Token + positional embeddings
                         |
                         v
              +-------------------------+
              | Transformer Block 0     |
              |                         |
              | LayerNorm               |
              | Multi-Head Self-Attention|
              | Residual Connection      |
              | LayerNorm               |
              | Feed-Forward MLP         |
              | Residual Connection      |
              +-------------------------+
                         |
                         v
              +-------------------------+
              | Transformer Block 1     |
              |                         |
              | LayerNorm               |
              | Multi-Head Self-Attention|
              | Residual Connection      |
              | LayerNorm               |
              | Feed-Forward MLP         |
              | Residual Connection      |
              +-------------------------+
                         |
                         v
                   Final LayerNorm
                         |
                         v
                      LM Head
                         |
                         v
                  Logits / Probabilities
                         |
                         v
                 Observatory Interface
```

The trained GPT configuration used for the BPE experiments has:

| Parameter | Value |
|---|---:|
| Architecture | Decoder-only GPT |
| Transformer layers | 2 |
| Attention heads per layer | 4 |
| Embedding dimension | 64 |
| Context length | 128 tokens |
| MLP hidden dimension | 256 |
| Tokenizer | Character-level and custom BPE |
| Training corpus | Tiny Shakespeare |

The custom BPE implementation is a learning-oriented character-level BPE baseline, rather than a full GPT-2-style byte-level tokenizer.

## Experiments

### Experiment 01 — Character Tokenization vs BPE

**Objective:** Measure token compression and compare the trained models using validation cross-entropy.

The corpus was split chronologically into 90% training data and 10% validation data. BPE merges were learned from the training portion.

| Metric | Character tokenizer | BPE tokenizer |
|---|---:|---:|
| Vocabulary size | 65 | 565 |
| Training tokens | 1,003,854 | 440,586 |
| Validation tokens | 111,540 | 50,475 |
| Validation loss | 1.7555 | 3.8542 |

**Key findings**

- BPE reduced the training token count by approximately **56.11%**.
- The BPE tokenizer achieved approximately **2.278 characters per token** on the training corpus.
- The raw validation losses are not directly comparable because the tokenizers define different prediction units and vocabulary sizes.

Validation loss was estimated using 100 randomly sampled batches, so results can vary between evaluations.

### Experiment 02 — Attention Behavior

**Objective:** Quantify how attention distributions differ across Transformer layers and input prompts.

Metrics:

- **Attention entropy:** measures how distributed attention weights are.
- **Effective attention positions:** \(e^H\), an entropy-derived measure of effective distribution width.
- **Mean backward distance:** the average distance into the past, weighted by attention probabilities.

Three prompts were evaluated using the trained BPE model.

| Prompt | Tokens |
|---|---:|
| Shakespeare question | 15 |
| Romeo dialogue | 23 |
| King dialogue | 24 |

**Observed pattern:** Layer 0 generally exhibited higher attention entropy and greater backward distance than Layer 1. The second layer tended to concentrate attention over fewer positions.

These measurements describe attention distributions; they do not establish specific semantic functions for individual heads.

### Experiment 03 — Attention Visualization

**Objective:** Visually inspect attention matrices across layers and heads.

The experiment generated heatmaps for all four heads in each of the two Transformer layers, across the three prompts.

The resulting six visualizations make it possible to inspect:

- Concentrated versus diffuse attention
- Attention to nearby versus earlier tokens
- Differences between heads and layers
- Causal masking patterns

Visual patterns are observations about the model's computations, not proof that an individual head performs a particular linguistic task.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| NumPy | From-scratch Transformer components |
| PyTorch | Model implementation, training and inference |
| Streamlit | Interactive inspection interface |
| Matplotlib / Seaborn | Attention visualizations |
| Pandas | Data handling and analysis |
| pytest | Automated testing |

## Project Structure

```text
transformer-observatory/
├── legacy_numpy/              # NumPy implementation
├── transformer/
│   ├── tokenizer.py            # Character tokenizer
│   ├── bpe.py                  # BPE tokenizer
│   ├── embeddings.py           # Token and positional embeddings
│   ├── attention.py            # Multi-head self-attention
│   ├── normalization.py        # Layer normalization
│   ├── mlp.py                  # Feed-forward network
│   ├── transformer_block.py    # Transformer block
│   ├── lm_head.py              # Language-model output head
│   ├── model.py                # GPT model
│   ├── batching.py             # Training batches
│   ├── observatory.py          # Model inspection
│   ├── visualization.py        # Visualization utilities
│   └── analysis.py              # Analysis utilities
├── experiments/
│   ├── compare_tokenizers.py
│   ├── attention_behavior.py
│   ├── attention_visualization.py
│   └── results/
├── checkpoints/
│   ├── best_gpt.pt
│   ├── best_gpt_bpe.pt
│   └── bpe_tokeniser.json
├── tests/
├── train.py
├── generate.py
├── app.py
└── README.md
```

*The tree describes the intended organization; adjust filenames to match the actual repository before publishing.*

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/transformer-observatory.git
cd transformer-observatory
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install the packages required by the project:

```bash
pip install torch numpy pandas matplotlib seaborn streamlit pytest
```

If the repository includes a pinned `requirements.txt`, prefer:

```bash
pip install -r requirements.txt
```

### 4. Prepare model assets

The Observatory requires a compatible trained checkpoint and its corresponding tokenizer.

Expected paths:

```text
checkpoints/best_gpt_bpe.pt
checkpoints/bpe_tokeniser.json
```

If checkpoints and data are excluded from version control, train the model locally or download the assets from the project's documented release location.

### 5. Launch the Observatory

From the repository root:

```bash
streamlit run app.py
```

### 6. Run experiments

Character vs BPE comparison:

```bash
python experiments/compare_tokenizers.py
```

Attention behavior:

```bash
python experiments/attention_behavior.py
```

Attention visualization:

```bash
python experiments/attention_visualization.py
```

Run automated tests:

```bash
pytest
```

## Limitations

- The model is intentionally small and trained on a limited literary corpus.
- Generated text is not expected to match the capabilities of modern large language models.
- The custom BPE implementation prioritizes learning and transparency over production tokenizer compatibility.
- Attention weights and activation statistics are useful inspection tools, but they do not independently establish causal explanations for model behavior.
- Quantitative results are based on a limited set of prompts and experiments.

## Project Status

**V1 — Complete**

The core implementation, training pipeline, inspection interface, and three experiments have been completed and verified.

Further work is intentionally out of scope for this version.

## Learning Objectives

This project was built to develop a practical understanding of:

- How Transformer computations fit together
- Why causal masking is required for autoregressive language modeling
- How tokenization affects sequence length and prediction loss
- How to expose intermediate neural-network computations
- How to measure and visualize attention behavior
- How to build reproducible experiments around a trained model

## Acknowledgements

Inspired by educational resources and explanations from Andrej Karpathy, Jay Alammar, and the original Transformer paper, *Attention Is All You Need*.

---

*Built as a hands-on exploration of Transformer internals, from mathematical foundations to an interactive inspection workbench.*
