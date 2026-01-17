# RNN From Scratch

This repository implements a **character-level vanilla Recurrent Neural Network (RNN) language model from scratch**, with an emphasis on **understanding basic recurrent sequence modeling through explicit implementation**.

All core components of the RNN are written manually, including hidden-state updates, without using high-level recurrent abstractions such as `nn.RNN`.

---

## Overview

The model is trained as a **character-level autoregressive language model** on the **WikiText-2 (raw)** dataset.
Given a sequence of characters, the model learns to predict the next character by propagating a hidden state sequentially through time.

The implementation focuses on:

* Explicit recurrence equations
* Sequential hidden-state propagation
* Clear and debuggable training behavior

---

## Model Architecture

The model follows a standard stacked vanilla RNN language model:

```
tokens
 → token embedding
 → N × RNN layers (from scratch)
 → linear output head
 → next-character prediction
```

### RNN Cell

Each RNN cell explicitly implements:

1. Linear transformation of input and hidden state
2. Nonlinear activation (tanh)

At each time step, the hidden state is updated and propagated through layers.
No gating or separate memory cell is used.

---

## Features

### Core Components

* From-scratch RNN cell (no `nn.RNN`)
* Explicit hidden-state update equation
* Explicit time-step loop
* Multi-layer RNN stack
* Linear output projection

### Training Pipeline

* Character-level dataset loader
* Sliding-window sequence generation
* Teacher forcing
* Cross-entropy loss
* Adam optimizer
* Gradient clipping
* Learning-rate scheduling
* Validation loop
* Checkpoint saving
* Text generation
* Training and validation loss plotting

### Platform Support

* Apple Silicon (MPS)
* CPU fallback
* Automatic device selection

---

## Repository Structure

```
rnn-from-scratch/
├── train.py
├── requirements.txt
├── checkpoints/
├── loss_curve.png
├── data/
│   └── wikitext-2-raw/
│       ├── wiki.train.raw
│       ├── wiki.valid.raw
│       └── wiki.test.raw
├── src/
│   ├── dataset.py
│   └── model/
│       ├── rnn_cell.py
│       └── rnn_model.py
└── README.md
```

---

## Dataset

The model is trained on **WikiText-2 (raw version)** using **character-level language modeling**.

* No tokenization is used
* Each training example is a fixed-length character sequence
* Targets are the same sequence shifted by one character

Approximate dataset sizes:

* Training: ~12M characters
* Validation: ~1M characters
* Test: ~1M characters

---

## Requirements

### System Requirements

* macOS, Linux, or Windows
* Apple Silicon supported via MPS
* CPU-only execution supported

### Software Requirements

* **Python ≥ 3.10**

All Python dependencies are listed in `requirements.txt`.

---

## Installation

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Download

The dataset is downloaded using the Hugging Face `datasets` API and saved locally as plain text.

```python
from datasets import load_dataset
import os

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
os.makedirs("data/wikitext-2-raw", exist_ok=True)

with open("data/wikitext-2-raw/wiki.train.raw", "w") as f:
    f.write("\n".join(dataset["train"]["text"]))

with open("data/wikitext-2-raw/wiki.valid.raw", "w") as f:
    f.write("\n".join(dataset["validation"]["text"]))

with open("data/wikitext-2-raw/wiki.test.raw", "w") as f:
    f.write("\n".join(dataset["test"]["text"]))
```

---

## Training

Run training from the repository root:

```bash
python train.py
```

During training, the script:

* Prints batch-level progress
* Reports training and validation loss per epoch
* Saves model checkpoints
* Generates sample text after each epoch
* Saves a training/validation loss plot (`loss_curve.png`)

---

## Text Generation

Text generation is performed using:

* Autoregressive decoding
* Sequential hidden-state propagation
* Temperature-scaled multinomial sampling

Generated samples provide a qualitative check on learning and short-term dependency modeling.

---

## Design Philosophy

This implementation prioritizes:

* Simplicity over sophistication
* Explicit recurrence over abstraction
* Transparency of hidden-state updates
* Correctness over throughput

This model serves as a clear baseline for understanding the limitations of ungated recurrent networks.

---

## License

MIT License
